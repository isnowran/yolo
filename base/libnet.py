import conf

def refine(x, rsize):
    x = x.permute(0, 2, 3, 1)
    s = x.shape
    x = x.view(s[0], s[1], s[2], -1, rsize)

    lb = conf.loc_box
    x[:, :, :, :, lb:lb+2] = x[:, :, :, :, lb:lb+2].sigmoid()
    x[:, :, :, :, conf.loc_p] = x[:, :, :, :, conf.loc_p].sigmoid()

    return x
