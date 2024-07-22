import subprocess

GPU_ID = 0
QF_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# command = [
#     'CUDA_VISIBLE_DEVICES={}'.format(GPU_ID),
#     '/usr/local/bin/python3.9', 'train_compression.py',
#     '--JPEG_enable', '--freeze_model', '--fixed_alpha',
#     '--JPEG_alpha', '100', '--JPEG_learning_rate', '0.1',
#     '--weight_lum', '1.0', '--weight_chrom', '1.0',
#     '--num_workers', '16'
# ]

for QF in QF_list:
    command = [
        'CUDA_VISIBLE_DEVICES={}'.format(GPU_ID),
        '/usr/local/bin/python3.9', 'train_compression.py',
        '--JPEG_enable', '--freeze_model', '--fixed_alpha', '--QF_Inital',
        '--JPEG_alpha', '100', '--JPEG_learning_rate', '0.01',
        '--Q_initial_QF', str(QF),
        '--weight_lum', '0.001', '--weight_chrom', '0.001',
        '--num_workers', '16'
    ]
    
    subprocess.run(' '.join(command), shell=True)