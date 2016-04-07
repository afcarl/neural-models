from keras.layers.core import Layer
from keras.layers import containers
from keras.layers.convolutional import Convolution2D
from keras import backend as K



class CrossChannelNormalization(Layer):
    def __init__(self, alpha = 1e-4, k=2, beta=0.75, n=5, **kwargs):
        self.alpha=alpha
        self.k = k
        self.beta = beta
        self.n = n

        super(CrossChannelNormalization, self).__init__(**kwargs)


    @property
    def output_shape(self):
        input_shape = self.input_shape
        return input_shape

    def get_output(self, train=False):
        X = self.get_input(train)
        b, ch, r, c = X.shape #self.input_shape
        #pdb.set_trace()
        half = self.n // 2
        square = K.square(X)

        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        # extra_channels = K.zeros((b, ch+2*half, r, c))
        # extra_channels[:,half:half+ch,:,:] = square
        
        scale = self.k
        for i in xrange(self.n):
            scale += self.alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** self.beta

        return X / scale
    
class SplitTensor(Layer):
    '''Repeat the input n times.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
    # Output shape
        4D tensor with shape:
        `(samples, selected_channels, rows, cols)`
    # Arguments
        axis: integer, axis of the selection.
        ratio_split: int, number of parts dividing the input
        id_split: int, number of the selected part
    '''
    def __init__(self, axis=1, ratio_split=1, id_split = 0, **kwargs):
        super(SplitTensor, self).__init__(**kwargs)
        self.axis=axis
        self.ratio_split = ratio_split
        self.id_split = id_split

    @property
    def output_shape(self):
        input_shape = self.input_shape
        output_shape = list(input_shape)
        output_shape[self.axis] = output_shape[self.axis] / self.ratio_split
        return tuple(output_shape)

    def get_output(self, train=False):
        X = self.get_input(train)
        id_split = self.id_split
        ratio_split = self.ratio_split
        div = self.input_shape[self.axis] / self.ratio_split
        
        axis=self.axis
        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output == X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")
        
        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'axis': self.axis,
                  'ratio_split': self.ratio_split,
                  'id_split': self.id_split}
        base_config = super(SplitTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





def Convolution2DGroup(n_group, nb_filter, nb_row, nb_col, input_shape, **kwargs):
    layer = containers.Graph()
    layer.name = kwargs['name']
    new_kwargs = dict((key, val) for key,val in kwargs.iteritems() if key != "name")
    layer.add_input(name='input', input_shape=input_shape[1:])
    for i in range(n_group):
        layer.add_node(SplitTensor(axis=1,ratio_split=n_group,id_split=i),
                       name='split'+str(i),
                       input='input')
        layer.add_node(Convolution2D(nb_filter / n_group,
                                     nb_row,nb_col,**new_kwargs),
                       name='conv'+str(i),
                       input='split'+str(i))

    layer.add_output(name='output',
                     inputs=['conv'+str(i) for i in range(n_group)],
                     merge_mode='concat',
                     concat_axis=1)
    

    return layer
        
