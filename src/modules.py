import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class PhysicalModel(object): # defines how the image projects to the CSI measurements

    def __init__(self, used_channel, transmit_scale):
        self.used_channel = used_channel
        self.transmit_scale = transmit_scale
        self.wavelen = used_channel[6]
        self.RIS_RCS = (4 * math.pi * (used_channel[7][0] * used_channel[7][1]).pow(2) / (used_channel[6]).pow(2)).pow(0.5)
        self.voxel_RCS = (4 * math.pi * (used_channel[8][0] * used_channel[8][1]).pow(2) / (used_channel[6]).pow(2)).pow(0.5)
        # the above RCS is sqrt RCS in fact, cite: 
        # Y. Huang, J. Yang, W. Tang, C.-K. Wen, S. Xia, and S. Jin, “Joint localization and environment 
        # sensing by harnessing NLOS components in RIS-aided mmWave communication systems,” IEEE Trans. Wireless Commun., 
        # vol. 22, no. 12, pp. 8797–8813, Dec. 2023.

    def illuminate(self, x, RIS_phase_power):

        if len(RIS_phase_power.shape) == 2: RIS_phase_power = RIS_phase_power.unsqueeze(dim=1)
        com = torch.complex(torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
        mea = torch.zeros(x.shape[0], 1, 2 * self.used_channel[0].shape[0] * self.used_channel[2].shape[1])
        for i in range(x.shape[0]):
            RIS_phase = torch.exp( - 1 * com * RIS_phase_power[i, :])
            h_tx_ris_rx = torch.mm(self.used_channel[0] * RIS_phase.repeat(self.used_channel[0].shape[0], 1) * self.RIS_RCS, self.used_channel[2])
            h_tx_roi_rx = torch.mm(self.used_channel[1] * x[i].repeat(self.used_channel[1].shape[0], 1) * self.voxel_RCS, self.used_channel[3])
            h_tx_ris_roi_rx = torch.mm(torch.mm(self.used_channel[0] * RIS_phase.repeat(self.used_channel[0].shape[0], 1) * self.RIS_RCS, 
                                                self.used_channel[4].t()) * x[i].repeat(self.used_channel[0].shape[0], 1) * self.voxel_RCS, 
                                                self.used_channel[3])
            h_tx_roi_ris_rx = torch.mm(torch.mm(self.used_channel[1] * x[i].repeat(self.used_channel[1].shape[0], 1) * self.voxel_RCS, 
                                                self.used_channel[4]) * RIS_phase.repeat(self.used_channel[1].shape[0], 1) * self.RIS_RCS, 
                                                self.used_channel[2])
            mea_in = h_tx_ris_rx + h_tx_roi_rx + h_tx_ris_roi_rx + h_tx_roi_ris_rx + self.used_channel[5]
            mea_in = mea_in.view(1, -1)
            mea_vec = torch.cat((mea_in.real, mea_in.imag), 1) * self.transmit_scale # complex measurements to real vectors
            mea[i, 0, :] = mea_vec

        mea = mea.to(device=x.device)
        return mea


class MeasureEmbeddingNetwork(nn.Module): # `g_t = relu( fc( fc(measurements) ) + fc( fc(RIS_phase_power) ) )`

    def __init__(self, measure_embedding_hidden_size, RIS_phase_power_embedding_hidden_size, learned_start, 
                 used_channel, transmit_scale):
        super(MeasureEmbeddingNetwork, self).__init__()
        self.physical = PhysicalModel(used_channel, transmit_scale)
        self.learned_start = learned_start
        if learned_start:
            first_RIS_phase_power = torch.rand(1, used_channel[0].shape[1]) * 2 * math.pi
            self.first_RIS_phase_power = Parameter(first_RIS_phase_power)
        
        self.input_dim_mea = 2 * used_channel[0].shape[0] * used_channel[2].shape[1]
        self.input_dim_RIS_phase = used_channel[0].shape[1] # directly processing RIS phase power (real values)

        self.model_measure_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_mea, measure_embedding_hidden_size[0]),
            torch.nn.BatchNorm1d(1, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(measure_embedding_hidden_size[0], measure_embedding_hidden_size[1]),
            torch.nn.BatchNorm1d(1, track_running_stats=False)
        )
        self.model_RIS_phase_power_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim_RIS_phase, RIS_phase_power_embedding_hidden_size[0]),
            torch.nn.BatchNorm1d(1, track_running_stats=False),
            torch.nn.ReLU(),
            torch.nn.Linear(RIS_phase_power_embedding_hidden_size[0], RIS_phase_power_embedding_hidden_size[1]),
            torch.nn.BatchNorm1d(1, track_running_stats=False)
        )

    def forward(self, x, RIS_phase_power): # generate single measurement (dim=2) from image x
        if RIS_phase_power is None and self.learned_start:
            RIS_phase_power = self.first_RIS_phase_power.repeat(x.shape[0], 1).unsqueeze(dim=1)
            single_measurement = self.physical.illuminate(x, RIS_phase_power)
        else:
            single_measurement = self.physical.illuminate(x, RIS_phase_power)

        single_measurement_embedding = self.model_measure_embedding(single_measurement)
        RIS_phase_power_embedding = self.model_RIS_phase_power_embedding(RIS_phase_power)
        measure_embedding = F.relu(single_measurement_embedding + RIS_phase_power_embedding)
        
        return measure_embedding
    

class ClassifyNetwork(nn.Module): # Uses the internal state `h_t` of the core network to produce the final output classification.

    def __init__(self, input_size, hidden_size, output_size):
        super(ClassifyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, state_now):
        probs = torch.squeeze(self.model(state_now))
        classifier_log_prob = torch.log(probs + 1e-10) # use 1e-10 to avoid NaN
        return classifier_log_prob


class RISPhaseCustomizationNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RISPhaseCustomizationNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(1, track_running_stats=False),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh() # adjust outputs to interval [-1, 1]
        )

    def forward(self, state_now):
        RIS_phase_power_next = self.model(state_now) * math.pi # adjust outputs to interval [-pi, pi]
        return RIS_phase_power_next

