from yowoformer.model.backbone3d import resnext, resnet, mobilenet, mobilenetv2, shufflenet, shufflenetv2, i3d

def build_backbone3D(config):
    backbone_3D = config['backbone3D']

    if backbone_3D == 'resnext101':
        backbone3D = resnext.resnext101(config)
    elif backbone_3D == 'resnet':
        backbone3D = resnet.build_resnet(config)
    elif backbone_3D == 'mobilenet':
        backbone3D = mobilenet.build_mobilenet(config)
    elif backbone_3D == 'mobilenetv2':
        backbone3D = mobilenetv2.build_mobilenetv2(config)
    elif backbone_3D == 'shufflenet':
        backbone3D = shufflenet.build_shufflenet(config)
    elif backbone_3D == 'shufflenetv2':
        backbone3D = shufflenetv2.build_shufflenetv2(config)
    elif backbone_3D == 'i3d':
        backbone3D = i3d.build_i3d(config)
    elif backbone_3D == 'videomae':
        # Check if using advanced version with specific method
        backbone_config = config.get('BACKBONE3D', {})
        method = backbone_config.get('METHOD', 'simple')

        if method in ['token', 'cross', 'hybrid']:
            # Use advanced VideoMAE with specific reconstruction method
            from yowoformer.model.backbone3d.videomae_advanced import build_videomae_advanced
            backbone3D = build_videomae_advanced(config)
            print(f"[build_backbone3D] Using VideoMAE-Advanced with {method.upper()} method")
        else:
            # Use simple VideoMAE (original)
            from yowoformer.model.backbone3d.videomae import build_videomae_backbone
            backbone3D = build_videomae_backbone(config)
            print(f"[build_backbone3D] Using VideoMAE with simple projection")

    elif backbone_3D == 'videomae_8frame':
        # VideoMAE with true 8-frame input (reduced GFLOPs/increased FPS)
        backbone_config = config.get('BACKBONE3D', {})
        method = backbone_config.get('METHOD', 'simple')

        from yowoformer.model.backbone3d.videomae_8frame import build_videomae_8frame
        backbone3D = build_videomae_8frame(config)
        print(f"[build_backbone3D] Using VideoMAE-8Frame with {method.upper()} method")

    return backbone3D