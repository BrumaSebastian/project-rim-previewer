import { Component } from '@angular/core';
import { ImageInputComponent } from '../../components/image-input/image-input.component';
import { ImageInput } from '../../types/imageInput';

@Component({
  selector: 'app-home',
  imports: [ImageInputComponent],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss',
})
export class HomeComponent {
  ImageInput = ImageInput;
}
