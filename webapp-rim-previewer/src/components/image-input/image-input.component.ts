import { CommonModule } from '@angular/common';
import { Component, input } from '@angular/core';
import { ImageInput } from '../../types/imageInput';

@Component({
  selector: 'app-image-input',
  imports: [CommonModule],
  templateUrl: './image-input.component.html',
  styleUrl: './image-input.component.scss',
})
export class ImageInputComponent {
  selectText = input<string>('');
  descriptionText = input<string>('');
  type = input<ImageInput>(ImageInput.None);

  imagePreview: string | ArrayBuffer | null = null;
  errorMessage: string = '';

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;

    if (input.files && input.files.length > 0) {
      const file = input.files[0];

      if (!file.type.startsWith('image/jpeg')) {
        this.errorMessage = 'Only JPG files are allowed!';
        this.imagePreview = null;
        return;
      }

      this.errorMessage = '';

      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }
}
