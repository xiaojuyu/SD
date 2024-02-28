
def get_network_config(dataset_name, step = None, train_data_variance = None):
    network_config = {}
    if dataset_name == 'CALTECH101_ET':
        network_config['in_channel'] = 2
        network_config['out_channel'] = 2
        network_config['inplanes'] = 64
        network_config['expansion'] = 2
    elif dataset_name == 'MNIST_DVS':
        network_config['in_channel'] = 2
        network_config['out_channel'] = 2
        network_config['inplanes'] = 64
        network_config['expansion'] = 2
    elif dataset_name == 'MNIST':
        network_config['in_channel'] = 1
        network_config['out_channel'] = 1
        network_config['inplanes'] = 64
        network_config['expansion'] = 2
    return network_config

def get_diffusion_config(dataset_name, device, step = None, train_data_variance = None):
    network_config = {}
    if dataset_name == 'CALTECH101_ET':
        network_config['noise_steps'] = 1000
        network_config['beta_start'] = 1e-4
        network_config['beta_end'] = 0.02
        network_config['img_size'] = [32,2,128,128,step]
        network_config['device'] = device
    elif dataset_name == 'MNIST_DVS':
        network_config['noise_steps'] = 1000
        network_config['beta_start'] = 1e-4
        network_config['beta_end'] = 0.02
        network_config['img_size'] = [32,2,128,128,step]
        network_config['device'] = device
    elif dataset_name == 'MNIST':
        network_config['noise_steps'] = 1000
        network_config['beta_start'] = 1e-4
        network_config['beta_end'] = 0.02
        network_config['img_size'] = [32,1,28,28,step]
        network_config['device'] = device
    return network_config
