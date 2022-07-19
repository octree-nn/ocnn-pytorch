import os
import argparse
import wget
import zipfile
import ssl


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='download_point_clouds')
args = parser.parse_args()

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data/ae_shapenet')


# The following line is to deal with the error of "SSL: CERTIFICATE_VERIFY_FAILED"
# when using wget. (Ref: https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3)
ssl._create_default_https_context = ssl._create_unverified_context


def download_point_clouds():
  # download via wget
  if not os.path.exists(root_folder):
    os.makedirs(root_folder)
  url = 'https://www.dropbox.com/s/z2x0mw4ai18f855/ocnn_completion.zip?dl=1'
  filename = os.path.join(root_folder, 'ae_shapenet.zip')
  if not os.path.exists(filename):
    print('Download %s' % filename)
    wget.download(url, filename)

  # unzip
  print('Unzip %s' % filename)
  with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(root_folder)


def _convert_ply_to_numpy(prefix='shape'):
  ply_folder = os.path.join(root_folder, prefix + '.ply')
  npy_folder = os.path.join(root_folder, prefix + '.npy')

  folders = os.listdir(ply_folder)
  for folder in folders:
    src_folder = os.path.join(ply_folder, folder)
    des_folder = os.path.join(npy_folder, folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      if filename.endswith('.ply'):
        ply_filename = os.path.join(src_folder, filename)
        npy_filename = os.path.join(des_folder, filename)

        # load ply

        # save npy


def convert_ply_to_numpy():
  _convert_ply_to_numpy('shape')
  _convert_ply_to_numpy('test.scans')


def create_filelist():
  pass


if __name__ == '__main__':
  eval('%s()' % args.run)
