"""Stage 07 CARLA segmentation fine-tune config for PytorchAutoDrive.

This config expects a GTAV-style dataset layout at:
  ../../carla-perception-lab/data/processed/pad_finetune/
    images/
    labels/
    data_lists/{train,val}.txt

Masks are already converted to Cityscapes train IDs (0..18, 255), so no LabelMap
transform is applied here.
"""

dataset = dict(
    name="GTAV_Segmentation",
    image_set="train",
    root="../../carla-perception-lab/data/processed/pad_finetune",
)

train_augmentation = dict(
    name="Compose",
    transforms=[
        dict(name="ToTensor"),
        dict(
            name="Resize",
            size_image=(512, 1024),
            size_label=(512, 1024),
        ),
        dict(name="RandomTranslation", trans_h=2, trans_w=2),
        dict(name="RandomHorizontalFlip", flip_prob=0.5),
    ],
)

test_augmentation = dict(
    name="Compose",
    transforms=[
        dict(name="ToTensor"),
        dict(
            name="Resize",
            size_image=(512, 1024),
            size_label=(512, 1024),
        ),
    ],
)

loss = dict(
    name="WeightedCrossEntropyLoss",
    ignore_index=255,
    weight=[
        2.8149201869965,
        6.9850029945374,
        3.7890393733978,
        9.9428062438965,
        9.7702074050903,
        9.5110931396484,
        10.311357498169,
        10.026463508606,
        4.6323022842407,
        9.5608062744141,
        7.8698215484619,
        9.5168733596802,
        10.373730659485,
        6.6616044044495,
        10.260489463806,
        10.287888526917,
        10.289801597595,
        10.405355453491,
        10.138095855713,
    ],
)

optimizer = dict(
    name="torch_optimizer",
    torch_optim_class="Adam",
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4,
)

lr_scheduler = dict(
    name="epoch_poly_scheduler",
    epochs=10,
    power=0.9,
)

train = dict(
    exp_name="erfnet_carla_finetune_512x1024",
    workers=2,
    batch_size=2,
    checkpoint="../../carla-perception-lab/data/checkpoints/erfnet_cityscapes_512x1024_20200918.pt",
    world_size=0,
    dist_url="env://",
    device="cuda",
    val_num_steps=250,
    save_dir="../../carla-perception-lab/output/segmentation/checkpoints",
    num_epochs=10,
    collate_fn=None,
    input_size=(512, 1024),
    original_size=(512, 1024),
    num_classes=19,
    eval_classes=19,
    selector=None,
    encoder_only=False,
    encoder_size=None,
)

test = dict(
    exp_name="erfnet_carla_finetune_512x1024",
    workers=0,
    batch_size=1,
    checkpoint="../../carla-perception-lab/data/checkpoints/erfnet_cityscapes_512x1024_20200918.pt",
    device="cuda",
    save_dir="../../carla-perception-lab/output/segmentation/checkpoints",
    collate_fn=None,
    original_size=(512, 1024),
    num_classes=19,
    eval_classes=19,
    selector=None,
    encoder_only=False,
    encoder_size=None,
)

model = dict(
    name="ERFNet",
    num_classes=19,
    dropout_1=0.03,
    dropout_2=0.3,
    pretrained_weights="../../carla-perception-lab/data/checkpoints/erfnet_encoder_pretrained.pth.tar",
)
