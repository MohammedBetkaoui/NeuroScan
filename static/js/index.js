
        // Scroll progress indicator
        window.addEventListener('scroll', () => {
            const scrollTop = document.documentElement.scrollTop;
            const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
            const scrollProgress = (scrollTop / scrollHeight) * 100;
            document.querySelector('.scroll-indicator').style.transform = `scaleX(${scrollProgress / 100})`;
        });

        // Mobile menu functionality
        document.addEventListener('DOMContentLoaded', function() {
            const mobileMenuBtn = document.getElementById('mobile-menu-btn');
            const mobileMenu = document.getElementById('mobile-menu');
            const mobileNavLinks = document.querySelectorAll('.mobile-nav-link');
            
            console.log('Initializing mobile menu...', { mobileMenuBtn, mobileMenu });
            
            if (!mobileMenuBtn || !mobileMenu) {
                console.error('Mobile menu elements not found!');
                return;
            }
            
            let isMenuOpen = false;
            
            // Toggle mobile menu
            function toggleMobileMenu() {
                console.log('Toggling menu, current state:', isMenuOpen);
                isMenuOpen = !isMenuOpen;
                
                // Update data attribute for CSS
                mobileMenu.setAttribute('data-open', isMenuOpen.toString());
                
                if (isMenuOpen) {
                    document.body.style.overflow = 'hidden';
                    // Add a slight delay to start link animations
                    setTimeout(() => {
                        const links = mobileMenu.querySelectorAll('.mobile-nav-link');
                        links.forEach((link, index) => {
                            link.style.animationDelay = `${0.1 + (index * 0.05)}s`;
                        });
                    }, 100);
                    console.log('Menu opened');
                } else {
                    document.body.style.overflow = '';
                    console.log('Menu closed');
                }
                
                // Update button icon with animation
                updateMenuIcon();
                updateButtonState();
            }
            
            function updateMenuIcon() {
                const iconContainer = mobileMenuBtn.querySelector('div');
                if (isMenuOpen) {
                    iconContainer.innerHTML = `
                        <svg class="w-5 h-5 sm:w-6 sm:h-6 transition-all duration-300 rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    `;
                } else {
                    iconContainer.innerHTML = `
                        <svg class="w-5 h-5 sm:w-6 sm:h-6 transition-all duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
                        </svg>
                    `;
                }
            }
            
            function updateButtonState() {
                if (isMenuOpen) {
                    mobileMenuBtn.classList.add('bg-gradient-to-r', 'from-blue-100', 'to-purple-100', 'text-blue-700');
                    mobileMenuBtn.classList.remove('text-gray-600');
                } else {
                    mobileMenuBtn.classList.remove('bg-gradient-to-r', 'from-blue-100', 'to-purple-100', 'text-blue-700');
                    mobileMenuBtn.classList.add('text-gray-600');
                }
            }
            
            // Initialize menu icon
            updateMenuIcon();
            
            // Add click listener to mobile menu button
            mobileMenuBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Mobile menu button clicked');
                toggleMobileMenu();
            });

            // Close mobile menu when clicking on nav links
            mobileNavLinks.forEach(link => {
                link.addEventListener('click', () => {
                    console.log('Nav link clicked');
                    if (isMenuOpen) {
                        toggleMobileMenu();
                    }
                });
            });

            // Close mobile menu when clicking outside
            document.addEventListener('click', function(e) {
                if (isMenuOpen && !mobileMenu.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
                    console.log('Clicked outside, closing menu');
                    toggleMobileMenu();
                }
            });

            // Handle window resize
            window.addEventListener('resize', function() {
                if (window.innerWidth >= 1024 && isMenuOpen) {
                    console.log('Window resized to desktop, closing menu');
                    toggleMobileMenu();
                }
            });

            // Prevent menu close when clicking inside menu
            mobileMenu.addEventListener('click', function(e) {
                e.stopPropagation();
            });
            
            // Add debug info
            console.log('Mobile menu initialized successfully');
        });

        // Active navigation link tracking
        function updateActiveNavLink() {
            const sections = document.querySelectorAll('section[id]');
            const navLinks = document.querySelectorAll('.nav-link');
            const mobileNavLinks = document.querySelectorAll('.mobile-nav-link');
            
            let current = '';
            
            sections.forEach(section => {
                const sectionTop = section.offsetTop - 100;
                const sectionHeight = section.offsetHeight;
                
                if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                    current = section.getAttribute('id');
                }
            });

            // Update desktop nav
            navLinks.forEach(link => {
                link.classList.remove('active', 'bg-blue-100', 'text-blue-700');
                link.classList.add('text-gray-600');
                
                if (link.getAttribute('href') === `#${current}`) {
                    link.classList.remove('text-gray-600');
                    link.classList.add('active', 'bg-blue-100', 'text-blue-700');
                }
            });

            // Update mobile nav
            mobileNavLinks.forEach(link => {
                link.classList.remove('active');
                
                if (link.getAttribute('href') === `#${current}`) {
                    link.classList.add('active');
                }
            });
        }

        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Header background and active link updates on scroll
        window.addEventListener('scroll', () => {
            const header = document.querySelector('header');
            
            // Update header background
            if (window.scrollY > 100) {
                header.classList.remove('bg-white/95');
                header.classList.add('bg-white/98', 'shadow-xl', 'border-b-2');
            } else {
                header.classList.remove('bg-white/98', 'shadow-xl', 'border-b-2');
                header.classList.add('bg-white/95');
            }
            
            // Update active navigation link
            updateActiveNavLink();
        });

        // Initialize active link on page load
        document.addEventListener('DOMContentLoaded', updateActiveNavLink);
   