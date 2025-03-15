import { Component } from '@angular/core';
import { ThemeChangerComponent } from '../theme-changer/theme-changer.component';

@Component({
  selector: 'app-header',
  imports: [ThemeChangerComponent],
  templateUrl: './header.component.html',
  styleUrl: './header.component.scss',
})
export class HeaderComponent {}
