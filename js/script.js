document.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('nav');
  const navList = document.querySelector('nav ul');

  if (!nav || !navList) return;

  const menuButton = document.createElement('button');
  menuButton.type = 'button';
  menuButton.className = 'menu-toggle';
  menuButton.setAttribute('aria-label', 'Ouvrir le menu');
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.innerHTML = '<span></span><span></span><span></span>';

  nav.appendChild(menuButton);

  const closeMenu = () => {
    navList.classList.remove('is-open');
    menuButton.setAttribute('aria-expanded', 'false');
    menuButton.setAttribute('aria-label', 'Ouvrir le menu');
  };

  menuButton.addEventListener('click', () => {
    const isOpen = navList.classList.toggle('is-open');
    menuButton.setAttribute('aria-expanded', String(isOpen));
    menuButton.setAttribute('aria-label', isOpen ? 'Fermer le menu' : 'Ouvrir le menu');
  });

  navList.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', closeMenu);
  });

  window.addEventListener('resize', () => {
    if (window.innerWidth > 768) closeMenu();
  });
});
