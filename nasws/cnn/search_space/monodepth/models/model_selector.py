from models import MidasNet, PlanarNet, DeepLab


def get_model(params):

    if params["name"] == "midas":
        model = MidasNet(params["checkpoint"], backbone=params["backbone"])
    elif params["name"] == "planar":
        model = PlanarNet(params["checkpoint"])
    elif params["name"] == "deeplab":
        model = DeepLab(backbone=params["backbone"], output_stride=16, sync_bn=False)
    else:
        print("Model {} not implemented.".format(params["name"]))
        model = None
        assert False

    return model
