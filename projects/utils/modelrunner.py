import numpy as np
import time
import pdb


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


def run(data=None, functions=None, CONFIG=None):

	train_err = {}
	valid_err = {}
	valid_acc = {}

	bookkeeping = {}
	bookkeeping['train_err'] = []
	bookkeeping['valid_err'] = []

	X_train, y_train = data['train']
	X_val, y_val = data['valid']
	X_test, y_test = data['test']


	for epoch in range(CONFIG['num_epochs']):
		# In each epoch, we do a full pass over the training data:
		loss = 0
		train_batches = 0
		start_time = time.time()


		for seqlength in X_train.keys():
			for batch in iterate_minibatches(X_train[seqlength], y_train[seqlength], CONFIG['batch_size'], shuffle=CONFIG['shuffle']):
				inputs, targets= batch
				loss += functions['train'](inputs, targets)
				train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for seqlength in X_val.keys():
			for batch in iterate_minibatches(X_val[seqlength], y_val[seqlength], CONFIG['batch_size'], shuffle=CONFIG['shuffle']):
				inputs, targets = batch
				err = functions['val_fn'](inputs, targets)
				val_err += err
#				val_acc += acc
				val_batches += 1

		bookkeeping['train_err'].append(loss)
		bookkeeping['valid_err'].append(val_err)


		print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, CONFIG['num_epochs'], time.time() - start_time))
		print("  Training loss:\t\t\t{:.6f}".format(loss / train_batches))
		print("  Validation loss:\t\t\t{:.6f}".format(val_err / val_batches))

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for seqlength in X_test.keys():
		for batch in iterate_minibatches(X_test[seqlength], y_test[seqlength], CONFIG['batch_size'], shuffle=CONFIG['shuffle']):
			inputs, targets = batch
			err = functions['val_fn'](inputs, targets)
			test_err += err
	#		test_acc += acc
			test_batches += 1

	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
			test_acc / test_batches * 100))

	return train_err, valid_err, valid_acc