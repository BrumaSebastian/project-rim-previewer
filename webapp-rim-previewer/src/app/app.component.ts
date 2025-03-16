import { Component, HostListener } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HeaderComponent } from '../components/header/header.component';
import { FooterComponent } from '../components/footer/footer.component';
import { NavigationMenuComponent } from "../components/navigation-menu/navigation-menu.component";

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, HeaderComponent, FooterComponent, NavigationMenuComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent {
  title = 'webapp-rim-previewer';
  isFixed = false;
  showHeader = false;

  @HostListener('window:scroll', [])
  onWindowScroll() {
    if (window.scrollY > 50) {
      if (!this.isFixed) {
        this.isFixed = true;
        setTimeout(() => (this.showHeader = true), 50); // Small delay for smooth transition
      }
    } else {
      this.showHeader = false;
      setTimeout(() => (this.isFixed = false), 300); // Wait for animation before removing fixed
    }
  }
}
