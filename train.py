from transformers import AutoImageProcessor, DeformableDetrForObjectDetection, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from PIL import Image

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
        'labels': [item['labels'] for item in batch]
    }

def main():
    # Load your dataset with trust_remote_code=True
    dataset = load_dataset('flir_dataset.py', trust_remote_code=True)

    # Load processor and model
    processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")

    # Preprocess the dataset
    def preprocess_data(examples):
        images = [Image.open(path).convert("RGB") for path in examples['image_path']]
        annotations = examples['annotations']
        
        targets = []
        for anno in annotations:
            target = {}
            target['boxes'] = [obj['bbox'] for obj in anno]
            target['labels'] = [obj['category_id'] for obj in anno]
            targets.append(target)

        inputs = processor(images=images, annotations=targets, return_tensors="pt")
        inputs['labels'] = targets
        return inputs

    train_dataset = dataset['train'].map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
    val_dataset = dataset['validation'].map(preprocess_data, batched=True, remove_columns=dataset['validation'].column_names)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./deformable_detr_flir",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=1e-4,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Start training
    trainer.train()

    # Save model and processor
    model.save_pretrained("./deformable_detr_flir")
    processor.save_pretrained("./deformable_detr_flir")

if __name__ == "__main__":
    main()
