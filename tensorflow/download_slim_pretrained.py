import os
import tarfile
from six.moves import urllib # Python 2/3 compatibility

tensorflow_models_url = "http://download.tensorflow.org/models/"

encoder_dict = {
    'inception_v3': 'inception_v3_2016_08_28.tar.gz',
    'resnet_v1_101': 'resnet_v1_101_2016_08_28.tar.gz',
    'resnet_v2_101': 'resnet_v2_101_2017_04_14.tar.gz',
    'mobilenet_v1_1.0': 'mobilenet_v1_1.0_224_2017_06_14.tar.gz'
}

def main(ckpt_dir):
    """
    1. Download Pretrained Encoder checkpoint as tarball
    2. Unzip it.
    """
    for encoder_name, ckpt_zip_name in encoder_dict.items():
        
        ckpt_url = tensorflow_models_url + ckpt_zip_name
        ckpt_zip_path = os.path.join(ckpt_dir, ckpt_zip_name)
        ckpt_path = os.path.join(ckpt_dir, encoder_name, '.ckpt')
        
        encoder_name = encoder_name.replace('_', ' ').title()
        
        if os.path.exists(ckpt_path):
            print(f'{encoder_name} already exists.')
        else:
            print(f'Downloading {encoder_name} checkpoint...')

            if not os.path.exists(ckpt_path):
                print('Downloading %s' % ckpt_zip_path)
                urllib.request.urlretrieve(ckpt_url, ckpt_zip_path)
                print('Done!')

                # Unzip checkpoint from tarball
                print('Extracting %s..' % ckpt_zip_path)
                tar = tarfile.open(ckpt_zip_path, 'r:gz')
                tar.extractall(path=ckpt_dir)
                tar.close()
                
                # Delete tarball
                os.remove(ckpt_zip_path)
                print('Done!')

            print(f'Checkpoint of {encoder_name} downloaded')

if __name__ == '__main__':
    # ckpt_dir need to be defined
    main(ckpt_dir)
