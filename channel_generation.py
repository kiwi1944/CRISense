import math
import torch
import logging


def channel_generation(config):

    logging.info(f'generating used channels')

    f = config.frequency * 1e9
    c = 3e8
    wavelen = c / f

    RIS_num2 = list(map(lambda x: int(x), config.RIS_num2.split(',')))
    RIS_num = RIS_num2[0] * RIS_num2[1]
    RIS_element_size = [wavelen / 2, wavelen / 2]

    dis = config.distance * wavelen # distance between RIS and ROI along x-axis
    ROI_element_size = [wavelen, wavelen, wavelen]
    ROI_num3 = list(map(lambda x: int(x), config.ROI_num3.split(',')))
    ROI_num = ROI_num3[0] * ROI_num3[1] * ROI_num3[2]

    BS_element_size = wavelen / 2
    Tx = list(map(lambda x: int(x), config.Tx.split(',')))
    Tx = torch.unsqueeze(torch.Tensor(Tx), dim=1) * wavelen
    Tx = Tx.repeat(1, config.antenna_num)
    Rx = list(map(lambda x: int(x), config.Rx.split(',')))
    Rx = torch.unsqueeze(torch.Tensor(Rx), dim=1) * wavelen
    Rx = Rx.repeat(1, config.antenna_num)

    # antenna direction is along y-axis
    tmp = torch.arange(0, config.antenna_num, 1) * BS_element_size
    Tx[1, :] = Tx[1, :] + tmp
    Rx[1, :] = Rx[1, :] + tmp

    # RIS element locations
    RIS_element_position_y = torch.arange(1, RIS_num2[0] + 1)
    RIS_element_position_y = (RIS_element_position_y - (RIS_element_position_y[0] + RIS_element_position_y[-1]) / 2) * RIS_element_size[0]
    RIS_element_position_z = torch.arange(1, RIS_num2[1] + 1)
    RIS_element_position_z = (RIS_element_position_z - (RIS_element_position_z[0] + RIS_element_position_z[-1]) / 2) * RIS_element_size[1]

    RIS_element_position = torch.zeros(3, RIS_num)
    RIS_element_position[1, :] = RIS_element_position_y.repeat_interleave(RIS_num2[1], dim=0)
    RIS_element_position[2, :] = RIS_element_position_z.repeat(1, RIS_num2[0])

    # ROI element locations
    ROI_element_position_x = torch.arange(1, ROI_num3[0] + 1)
    ROI_element_position_x = dis + (ROI_element_position_x - (ROI_element_position_x[0] + ROI_element_position_x[-1]) / 2) * ROI_element_size[0]
    ROI_element_position_y = torch.arange(1, ROI_num3[1] + 1)
    ROI_element_position_y = (ROI_element_position_y - (ROI_element_position_y[0] + ROI_element_position_y[-1]) / 2) * ROI_element_size[1]
    ROI_element_position_z = torch.arange(1, ROI_num3[2] + 1)
    ROI_element_position_z = (ROI_element_position_z - (ROI_element_position_z[0] + ROI_element_position_z[-1]) / 2) * ROI_element_size[2]

    ROI_element_position = torch.zeros(3, ROI_num)
    ROI_element_position[0, :] = ROI_element_position_x.repeat_interleave(ROI_num3[1] * ROI_num3[2], dim=0)
    ROI_element_position[1, :] = (ROI_element_position_y.repeat_interleave(ROI_num3[2], dim=0)).repeat(1,  ROI_num3[0])
    ROI_element_position[2, :] = ROI_element_position_z.repeat(1, ROI_num3[0] * ROI_num3[1])

    # obtain channel response and B
    h_tx_ris = construct_los_channel(wavelen, Tx, RIS_element_position)
    h_tx_roi = construct_los_channel(wavelen, Tx, ROI_element_position)
    h_ris_rx = construct_los_channel(wavelen, RIS_element_position, Rx)
    h_roi_rx = construct_los_channel(wavelen, ROI_element_position, Rx)
    h_roi_ris = construct_los_channel(wavelen, ROI_element_position, RIS_element_position)

    h_tx_rx = construct_los_channel(wavelen, Tx, Rx) * config.self_interference

    used_channel = [h_tx_ris, h_tx_roi, h_ris_rx, h_roi_rx, h_roi_ris, h_tx_rx, # 5 channels
                    torch.tensor(wavelen), torch.tensor(RIS_element_size), torch.tensor(ROI_element_size)] # 3, additionally used

    return used_channel


# calculate LOS channel, data type: all tensor
def construct_los_channel(wavelen, tx_pos, rx_pos):
    
    dim1 = tx_pos.shape[1]
    dim2 = rx_pos.shape[1]
    dis = torch.zeros(dim1, dim2)
    
    if dim1 >= dim2:
        for i in range(0, dim2):
            tmp = tx_pos - torch.unsqueeze(rx_pos[:, i], dim=1)
            dis[:, i] = torch.sum(tmp * tmp, dim=0) ** 0.5
    else:
        for i in range(0, dim1):
            tmp = rx_pos - torch.unsqueeze(tx_pos[:, i], dim=1)
            dis[i, :] = torch.sum(tmp * tmp, dim=0) ** 0.5
    
    com = torch.complex(torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
    h = (1 / ((4 * math.pi) ** 0.5)) / dis * torch.exp( - 1 * com * 2 * math.pi * dis / wavelen)
    
    return h
