import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# python3.8 ./JPEG_DNN_analysis/compare_Q_tables.py --dir "./save/CUB200/teacher_Q_tables/T:resnet18_lr_0.1_decay_0.0001_log_JPEG/"

plt.rcParams["font.family"] = "Times New Roman"
font = 25
# Set the default text font size
# plt.rc('font', size=14, weight='bold')
plt.rc('figure', figsize = (8,6))
# plt.rc('figure', autolayout = True)
plt.rc('font', size=font+4)
# Set the axes title font size
plt.rc('axes', titlesize=font)
# Set the axes labels font size
plt.rc('axes', labelsize=font)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=font)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=font)
# Set the legend font size
plt.rc('legend', fontsize=15)
# Set the font size of the figure title
plt.rc('figure', titlesize=font)


def zigzag(matrix):
    zigzag = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                           [2, 4, 7, 13, 16, 26, 29, 42],
                           [3, 8, 12, 17, 25, 30, 41, 43],
                           [9, 11, 18, 24, 31, 40, 44, 53],
                           [10, 19, 23, 32, 39, 45, 52, 54],
                           [20, 22, 33, 38, 46, 51, 55, 60],
                           [21, 34, 37, 47, 50, 56, 59, 61],
                           [35, 36, 48, 49, 57, 58, 62, 63]])
    
    

    # Get the shape of the matrix
    matrix_size = np.shape(matrix)

    # Calculate the size of the vector
    vector_size = np.shape(zigzag)

    # Check if the matrix size matches the vector size
    if matrix_size != vector_size:
        raise ValueError("The matrix size does not match the vector size.")

    # Create an empty vector to store the values
    vector = np.zeros(matrix_size[0] * matrix_size[1])

    # Iterate over each element in the matrix and place it in the corresponding position in the vector
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            vector[index] = matrix[i, j]

    return vector


def inverse_zigzag(vector):
    zigzag = torch.tensor([[0, 1, 5, 6, 14, 15, 27, 28],
                           [2, 4, 7, 13, 16, 26, 29, 42],
                           [3, 8, 12, 17, 25, 30, 41, 43],
                           [9, 11, 18, 24, 31, 40, 44, 53],
                           [10, 19, 23, 32, 39, 45, 52, 54],
                           [20, 22, 33, 38, 46, 51, 55, 60],
                           [21, 34, 37, 47, 50, 56, 59, 61],
                           [35, 36, 48, 49, 57, 58, 62, 63]])

    # Get the shape of the vector
    vector_shape = vector.size()

    # Calculate the size of the 2D matrix
    matrix_size = zigzag.size()

    # Check if the vector size matches the matrix size
    if vector_shape != matrix_size:
        raise ValueError("The vector size does not match the matrix size.")

    # Create an empty matrix to store the values
    matrix = torch.zeros(matrix_size)

    # Iterate over each element in the vector and place it in the corresponding position in the matrix
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            index = zigzag[i, j]
            matrix[i, j] = vector[index]

    return matrix

def get_average_model_magnitude(model):
    total_magnitude = 0.0
    num_parameters = 0
    for param in model.parameters():
        param_magnitude = torch.mean(torch.abs(param)).item()
        total_magnitude += param_magnitude
        num_parameters += 1
    if num_parameters == 0:
        return 0.0
    return total_magnitude / num_parameters

def get_max_model_magnitude(model):
    max_magnitude = 0.0
    for param in model.parameters():
        param_max = torch.max(torch.abs(param)).item()
        if param_max > max_magnitude:
            max_magnitude = param_max
    return max_magnitude

def quantizationTable(QF=50, Luminance=True):
    #  Luminance quantization table
    #  Standard
    # * 16 11 10 16 24  40  51  61
    # * 12 12 14 19 26  58  60  55
    # * 14 13 16 24 40  57  69  56
    # * 14 17 22 29 51  87  80  62
    # * 18 22 37 56 68  109 103 77
    # * 24 35 55 64 81  104 113 92
    # * 49 64 78 87 103 121 120 101
    # * 72 92 95 98 112 100 103 99

    quantizationTableData = np.ones((8, 8), dtype=np.float32)

    if QF == 100:
        # print(quantizationTableData)
        return quantizationTableData

    if Luminance == True:  # Y channel
        quantizationTableData[0][0] = 16
        quantizationTableData[0][1] = 11
        quantizationTableData[0][2] = 10
        quantizationTableData[0][3] = 16
        quantizationTableData[0][4] = 24
        quantizationTableData[0][5] = 40
        quantizationTableData[0][6] = 51
        quantizationTableData[0][7] = 61
        quantizationTableData[1][0] = 12
        quantizationTableData[1][1] = 12
        quantizationTableData[1][2] = 14
        quantizationTableData[1][3] = 19
        quantizationTableData[1][4] = 26
        quantizationTableData[1][5] = 58
        quantizationTableData[1][6] = 60
        quantizationTableData[1][7] = 55
        quantizationTableData[2][0] = 14
        quantizationTableData[2][1] = 13
        quantizationTableData[2][2] = 16
        quantizationTableData[2][3] = 24
        quantizationTableData[2][4] = 40
        quantizationTableData[2][5] = 57
        quantizationTableData[2][6] = 69
        quantizationTableData[2][7] = 56
        quantizationTableData[3][0] = 14
        quantizationTableData[3][1] = 17
        quantizationTableData[3][2] = 22
        quantizationTableData[3][3] = 29
        quantizationTableData[3][4] = 51
        quantizationTableData[3][5] = 87
        quantizationTableData[3][6] = 80
        quantizationTableData[3][7] = 62
        quantizationTableData[4][0] = 18
        quantizationTableData[4][1] = 22
        quantizationTableData[4][2] = 37
        quantizationTableData[4][3] = 56
        quantizationTableData[4][4] = 68
        quantizationTableData[4][5] = 109
        quantizationTableData[4][6] = 103
        quantizationTableData[4][7] = 77
        quantizationTableData[5][0] = 24
        quantizationTableData[5][1] = 35
        quantizationTableData[5][2] = 55
        quantizationTableData[5][3] = 64
        quantizationTableData[5][4] = 81
        quantizationTableData[5][5] = 104
        quantizationTableData[5][6] = 113
        quantizationTableData[5][7] = 92
        quantizationTableData[6][0] = 49
        quantizationTableData[6][1] = 64
        quantizationTableData[6][2] = 78
        quantizationTableData[6][3] = 87
        quantizationTableData[6][4] = 103
        quantizationTableData[6][5] = 121
        quantizationTableData[6][6] = 120
        quantizationTableData[6][7] = 101
        quantizationTableData[7][0] = 72
        quantizationTableData[7][1] = 92
        quantizationTableData[7][2] = 95
        quantizationTableData[7][3] = 98
        quantizationTableData[7][4] = 112
        quantizationTableData[7][5] = 100
        quantizationTableData[7][6] = 103
        quantizationTableData[7][7] = 99
    else:
        # Standard Cb Cr channel
        # 17 18  24  47  99  99  99  99
        # 18 21  26  66  99  99  99  99
        # 24 26  56  99  99  99  99  99
        # 47 66  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99
        # 99 99  99  99  99  99  99  99

        quantizationTableData[0][0] = 17
        quantizationTableData[0][1] = 18
        quantizationTableData[0][2] = 24
        quantizationTableData[0][3] = 47
        quantizationTableData[0][4] = 99
        quantizationTableData[0][5] = 99
        quantizationTableData[0][6] = 99
        quantizationTableData[0][7] = 99
        quantizationTableData[1][0] = 18
        quantizationTableData[1][1] = 21
        quantizationTableData[1][2] = 26
        quantizationTableData[1][3] = 66
        quantizationTableData[1][4] = 99
        quantizationTableData[1][5] = 99
        quantizationTableData[1][6] = 99
        quantizationTableData[1][7] = 99
        quantizationTableData[2][0] = 24
        quantizationTableData[2][1] = 26
        quantizationTableData[2][2] = 56
        quantizationTableData[2][3] = 99
        quantizationTableData[2][4] = 99
        quantizationTableData[2][5] = 99
        quantizationTableData[2][6] = 99
        quantizationTableData[2][7] = 99
        quantizationTableData[3][0] = 47
        quantizationTableData[3][1] = 66
        quantizationTableData[3][2] = 99
        quantizationTableData[3][3] = 99
        quantizationTableData[3][4] = 99
        quantizationTableData[3][5] = 99
        quantizationTableData[3][6] = 99
        quantizationTableData[3][7] = 99
        quantizationTableData[4][0] = 99
        quantizationTableData[4][1] = 99
        quantizationTableData[4][2] = 99
        quantizationTableData[4][3] = 99
        quantizationTableData[4][4] = 99
        quantizationTableData[4][5] = 99
        quantizationTableData[4][6] = 99
        quantizationTableData[4][7] = 99
        quantizationTableData[5][0] = 99
        quantizationTableData[5][1] = 99
        quantizationTableData[5][2] = 99
        quantizationTableData[5][3] = 99
        quantizationTableData[5][4] = 99
        quantizationTableData[5][5] = 99
        quantizationTableData[5][6] = 99
        quantizationTableData[5][7] = 99
        quantizationTableData[6][0] = 99
        quantizationTableData[6][1] = 99
        quantizationTableData[6][2] = 99
        quantizationTableData[6][3] = 99
        quantizationTableData[6][4] = 99
        quantizationTableData[6][5] = 99
        quantizationTableData[6][6] = 99
        quantizationTableData[6][7] = 99
        quantizationTableData[7][0] = 99
        quantizationTableData[7][1] = 99
        quantizationTableData[7][2] = 99
        quantizationTableData[7][3] = 99
        quantizationTableData[7][4] = 99
        quantizationTableData[7][5] = 99
        quantizationTableData[7][6] = 99
        quantizationTableData[7][7] = 99

    if QF >= 1:
        if QF < 50:
            S = 5000 / QF
        else:
            S = 200 - 2 * QF

        for i in range(8):
            for j in range(8):
                q = (50 + S * quantizationTableData[i][j]) / 100
                q = np.clip(np.floor(q), 1, 255)
                quantizationTableData[i][j] = q
    return quantizationTableData



def get_zigzag():
    zigzag = torch.tensor(( [[0,   1,   5,  6,   14,  15,  27,  28],
                             [2,   4,   7,  13,  16,  26,  29,  42],
                             [3,   8,  12,  17,  25,  30,  41,  43],
                             [9,   11, 18,  24,  31,  40,  44,  53],
                             [10,  19, 23,  32,  39,  45,  52,  54],
                             [20,  22, 33,  38,  46,  51,  55,  60],
                             [21,  34, 37,  47,  50,  56,  59,  61],
                             [35,  36, 48,  49,  57,  58,  62,  63]]))
    return zigzag

def plot_q(CHANNEL, QT_HUMAN, filename, channel_name):
    plt.figure()
    # plt.plot(np.arange(1, 65), plot_data(CHANNEL),  label="Model")
    # plt.plot(np.arange(1, 65), plot_data(QT_HUMAN),  label="Human")
    plt.plot(np.arange(1, 65), (CHANNEL),  label="Model")
    tick_positions = np.arange(1, 65, 15)  # Tick positions every 5 units
    tick_labels = np.arange(1, 65, 15)  # Corresponding tick labels
    plt.xticks(tick_positions, tick_labels)
    plt.grid(True, linestyle='dashed', linewidth=1)  # Customize the grid lines
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.xlabel("Frequency Index")
    plt.ylabel("Quantization Steps")
    plt.title('{} Channel'.format(channel_name))
    plt.savefig("{}_{}_channel.png".format(filename, channel_name),dpi=900, bbox_inches='tight')
    plt.close()

QT_Y = zigzag(quantizationTable(QF=50, Luminance=True))
QT_C = zigzag(quantizationTable(QF=50, Luminance=False))



parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--dir', type=str, default='./save/teacher_Q_tables/*/')
opt = parser.parse_args()

directory = opt.dir


# Using glob to load all files within the directory
files = glob.glob( directory + '*.pt')

plot_data = lambda x: x /x[0]

for file in files:
    print(file)
    plot_dir =  file.rsplit("/",1)[0]+"/plot_q/"
    filename = plot_dir + file.split("/")[-1].split(".pt")[0]
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    try:
        Q_table =  torch.load(file).cpu().detach().numpy()
    except:
        Q_table =  torch.load(file).detach().numpy()

    try:
        Y_channel = zigzag(Q_table[0])
        Cb_channel = zigzag(Q_table[1])
        Cr_channel = zigzag(Q_table[2])

        plot_q(Y_channel, QT_Y, filename, "Y")
        plot_q(Cb_channel, QT_C, filename, "Cb")
        plot_q(Cr_channel, QT_C, filename, "Cr")
    
    except:
        continue