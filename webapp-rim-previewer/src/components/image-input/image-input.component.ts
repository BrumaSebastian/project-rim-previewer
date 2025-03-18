import { CommonModule } from '@angular/common';
import { Component, inject, input } from '@angular/core';
import { ImageInput } from '../../types/imageInput';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-image-input',
  imports: [CommonModule],
  templateUrl: './image-input.component.html',
  styleUrl: './image-input.component.scss',
})
export class ImageInputComponent {
  http: HttpClient = inject(HttpClient);

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

      this.uploadImage(file);
    }
  }

  uploadImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);

    this.http
      .post<{ class: string; confidence: number }>(
        'http://127.0.0.1:8000/predict/',
        formData
      )
      .subscribe(
        (response) => {
          console.log('Prediction:', response);
        },
        (error) => {
          console.error('Error:', error);
        }
      );
  }
}
