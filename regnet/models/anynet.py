
import tensorflow as tf
# from reg_net_config  import cfg
from regnet.reg_net_config200MF  import cfg


class group_conv(tf.keras.layers.Layer):
    def __init__(self,feat_in,feat_out,kernel_size,strides,groups,use_bias=False):
        super(group_conv, self).__init__()
        self.groups = groups
        self.group_in_channel = int(feat_in/groups) # it needs to be int, fully devided
        assert self.group_in_channel * groups == feat_in

        self.group_out_channel = int(feat_out/groups) # it needs to be int, fully devided
        assert self.group_out_channel * groups == feat_out
        self.list_of_group_conv2d = []
        for _ in range(groups):
            self.list_of_group_conv2d.append(tf.keras.layers.Conv2D(self.group_out_channel,kernel_size = kernel_size,strides=strides,use_bias=False))
        

    def call(self,x):
        list_of_group_out = []
        for i in range(self.groups):
            gx = x[:,:,:,i:i+self.group_in_channel]
            gx = self.list_of_group_conv2d[i](gx)
            list_of_group_out.append(gx)
        
        x = tf.concat(list_of_group_out,axis=3)
        return x




def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    # stem_funs = {
    #     "res_stem_cifar": ResStemCifar,
    #     "res_stem_in": ResStemIN,
    #     "simple_stem_in": SimpleStemIN,
    # }
    stem_funs = {
        "simple_stem_in": SimpleStemIN
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]

class SimpleStemIN(tf.keras.Model):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, w_in, w_out):
        super(SimpleStemIN, self).__init__()
        # self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        # self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.padding_1 = tf.keras.layers.ZeroPadding2D((1,1))
        self.conv = tf.keras.layers.Conv2D(w_out, 3, strides=2, padding="valid", use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.padding_1(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ResStemIN(tf.keras.Model):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        super(ResStemIN, self).__init__()
        # self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False) # (Hin - 1)/2 + 1
        # self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        # self.pool = nn.MaxPool2d(3, stride=2, padding=1)  # (Hin)/2 + 1

        self.padding_3 = tf.keras.layers.ZeroPadding2D((3,3))
        self.conv = tf.keras.layers.Conv2D(w_out, 7, strides=2, padding="valid", use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.relu = tf.keras.layers.ReLU()
        self.padding_1 = tf.keras.layers.ZeroPadding2D((1,1))
        self.pool = tf.keras.layers.MaxPool2D(3, strides=2, padding="valid")

    def forward(self, x):
        x = self.padding_3(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.padding_1(x)
        x = self.pool(x)

        return x

def get_block_fun(block_type):
    """Retrieves the block function by name."""
    # block_funs = {
    #     "vanilla_block": VanillaBlock,
    #     "res_basic_block": ResBasicBlock,
    #     "res_bottleneck_block": ResBottleneckBlock,
    # }
    block_funs = {
        "res_bottleneck_block": ResBottleneckBlock
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(tf.keras.Model):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(w_in, nc, bias=True)
        # self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(nc, use_bias=True)
    
    def avg_pool(self,x):
        kerneal_size = x.shape[1:3] # h,w
        print("kerneal_size:",kerneal_size)
        x = tf.nn.avg_pool2d(x,kerneal_size,1,"VALID")
        return x

    def call(self, x):
        x = self.avg_pool(x)
        x = tf.reshape(x,[-1,self.w_in])
        x = self.fc(x)
        return x
    
    


class VanillaBlock(tf.keras.Model):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Vanilla block does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(VanillaBlock, self).__init__()
        # self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        # self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        # self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

        self.padding_1 = tf.keras.layers.ZeroPadding2D((1,1))
        self.a = tf.keras.layers.Conv2D(w_out, 3, strides=stride, padding="valid", use_bias=False)
        self.a_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = tf.keras.layers.ReLU()
        self.b = tf.keras.layers.Conv2D(w_out, 3, strides=1, padding="valid", use_bias=False)
        self.b_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.padding_1(x)
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.padding_1(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        return x



class BasicTransform(tf.keras.Model):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        # self.a = nn.Conv2d(w_in, w_out, 3, stride=stride, padding=1, bias=False)
        # self.a_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # self.b = nn.Conv2d(w_out, w_out, 3, stride=1, padding=1, bias=False)
        # self.b_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.b_bn.final_bn = True

        self.padding_1 = tf.keras.layers.ZeroPadding2D((1,1))
        self.a = tf.keras.layers.Conv2D(w_out, 3, strides=stride, padding="valid", use_bias=False)
        self.a_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = tf.keras.layers.ReLU()
        self.b = tf.keras.layers.Conv2D(w_out, 3, strides=1, padding="valid", use_bias=False)
        self.b_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_bn.final_bn = True # ?????


    def call(self, x):
        x = self.padding_1(x)
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.padding_1(x)
        x = self.b(x)
        x = self.b_bn(x)
       
        return x


class SE(tf.keras.Model):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.f_ex = nn.Sequential(
        #     nn.Conv2d(w_in, w_se, 1, bias=True),
        #     nn.ReLU(inplace=cfg.MEM.RELU_INPLACE),
        #     nn.Conv2d(w_se, w_in, 1, bias=True),
        #     nn.Sigmoid(),
        # )
        self.w_se = w_se
        self.w_in = w_in

        
        # self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        
        # self.f_ex = tf.keras.Sequential(
        #     [tf.keras.layers.Conv2D(self.w_se, 1, strides=1, padding="valid", use_bias=True),
        #     tf.keras.layers.ReLU(),
        #     tf.keras.layers.Conv2D(self.w_in, 1, strides=1, padding="valid", use_bias=True),
        #     tf.keras.layers.Activation("sigmoid")
        #     ]
        # ).build((None,1,1,w_in))
        self.f_ex = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(self.w_se, 1, strides=1, padding="valid", use_bias=True),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(self.w_in, 1, strides=1, padding="valid", use_bias=True),
            tf.keras.layers.Activation("sigmoid")
            ]
        )

    
    # def expand_hw(self,x):
    #     x = tf.expand_dims(x,1) # add w
    #     x = tf.expand_dims(x,1) # add h
    #     return x
    def avg_pool(self,x):
        kerneal_size = x.shape[1:3] # h,w
        print("kerneal_size:",kerneal_size)
        x = tf.nn.avg_pool2d(x,kerneal_size,1,"VALID")
        return x

    def call(self, x):

        return x * self.f_ex(self.avg_pool(x))




class BottleneckTransform(tf.keras.Model):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        # self.a = nn.Conv2d(w_in, w_b, 1, stride=1, padding=0, bias=False)
        # self.a_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # self.b = nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False)
        # self.b_bn = nn.BatchNorm2d(w_b, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        # if se_r:
        #     w_se = int(round(w_in * se_r))
        #     self.se = SE(w_b, w_se)
        # self.c = nn.Conv2d(w_b, w_out, 1, stride=1, padding=0, bias=False)
        # self.c_bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.c_bn.final_bn = True

        
        self.a = tf.keras.layers.Conv2D(w_b, 1, strides=1, padding="valid", use_bias=False)
        self.a_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.a_relu = tf.keras.layers.ReLU()
        self.padding_1 = tf.keras.layers.ZeroPadding2D((1,1))
        self.b = group_conv(feat_in = w_b,feat_out = w_b, kernel_size = 3, strides = stride, groups = g,use_bias=False)
        self.b_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.b_relu = tf.keras.layers.ReLU()
        self.se_r = se_r
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = tf.keras.layers.Conv2D(w_out, 1, strides=1, padding="valid", use_bias=False)
        self.c_bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.c_bn.final_bn = True

    def call(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.padding_1(x)
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)
        if self.se_r:
            x = self.se(x)
        x = self.c(x)
        x = self.c_bn(x)     
        return x

class ResBottleneckBlock(tf.keras.Model):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        # # Use skip connection with projection if shape changes
        # self.proj_block = (w_in != w_out) or (stride != 1)
        # if self.proj_block:
        #     self.proj = nn.Conv2d(w_in, w_out, 1, stride=stride, padding=0, bias=False)
        #     self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        # self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        # self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = tf.keras.layers.Conv2D(w_out, 1, strides=stride, padding="valid", use_bias=False)
            self.bn = tf.keras.layers.BatchNormalization(epsilon=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class AnyStage(tf.keras.Model):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self.block_list = []
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # name = "b{}".format(i + 1)
            # self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))
            self.block_list.append(block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def call(self, x):
        for block in self.block_list:
            x = block(x)
        return x




class AnyNet(tf.keras.Model):
    """AnyNet model."""

    @staticmethod
    def get_args():
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "ds": cfg.ANYNET.DEPTHS,
            "ws": cfg.ANYNET.WIDTHS,
            "ss": cfg.ANYNET.STRIDES,
            "bms": cfg.ANYNET.BOT_MULS,
            "gws": cfg.ANYNET.GROUP_WS,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else None,
            "nc": cfg.MODEL.NUM_CLASSES,
        }
    
    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        kwargs = self.get_args() if not kwargs else kwargs
        self._construct(**kwargs)
        # self.apply(net.init_weights) # no need to do this since ZERO_INIT_FINAL_GAMMA = False


    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
  
        
        self.stage_list = []
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            # self.add_module(name, AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            self.stage_list.append(AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            prev_w = w
          
        self.head = AnyHead(w_in=prev_w, nc=nc)
    
    def call(self, x):
        skips=[]
        x = self.stem(x)
        print(x)
        skips.append(x)
        for stage in self.stage_list:
            x = stage(x)
            print(x)
            skips.append(x)
        # x = self.head(x) # for conv hidden layer, do not flatten
        return skips

