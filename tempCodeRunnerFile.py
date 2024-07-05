def main():
    base_dir = 'Datasets'
    train_dataset = load_flir_dataset(
        os.path.join(base_dir, 'images_thermal_train', 'data'),
        os.path.join(base_dir, 'images_thermal_train', 'coco.json')
    )
    val_dataset = load_flir_dataset(
        os.path.join(base_dir, 'images_thermal_val', 'data'),
        os.path.join(base_dir, 'images_thermal_val', 'coco.json')