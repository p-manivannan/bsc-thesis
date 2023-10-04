import h5py

def save_data(filename, inputs, targets):
  file = h5py.File(filename + '.h5', 'w')
  file.create_dataset('inputs', data=inputs)
  file.create_dataset('targets', data=targets)
  file.close()


# Load only first 2 subjects for testing purposes
def load_data(filename):
  file = h5py.File('/home/pmanivannan/bsc-thesis/' + filename+'.h5', 'r')
  inputs = file['inputs'][0:1152]
  targets = file['targets'][:1152]
  file.close()
  return inputs, targets

