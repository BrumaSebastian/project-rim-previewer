import { Component } from '@angular/core';

@Component({
  selector: 'app-theme-changer',
  imports: [],
  templateUrl: './theme-changer.component.html',
  styleUrl: './theme-changer.component.scss',
})
export class ThemeChangerComponent {
  changeTheme(event: Event) {
    // const theme = (event.target as HTMLSelectElement).value;
    // console.log(theme);
    // document.documentElement.setAttribute('data-theme', theme);
  }
}
