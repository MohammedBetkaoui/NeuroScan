
        // Smooth scrolling pour les ancres (si pas nativement supporté)
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

        // Animation fade-in au scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, observerOptions);

        // Observer toutes les sections
        document.querySelectorAll('section').forEach(section => {
            observer.observe(section);
        });

        // Highlight du menu de navigation mobile
        function updateActiveNavItem() {
            const sections = ['fonctionnalites', 'utilisation', 'exemples', 'support'];
            const navLinks = document.querySelectorAll('nav a');
            
            sections.forEach((sectionId, index) => {
                const section = document.getElementById(sectionId);
                if (section) {
                    const rect = section.getBoundingClientRect();
                    if (rect.top <= 100 && rect.bottom >= 100) {
                        navLinks.forEach(link => link.classList.remove('bg-primary-600', 'text-white'));
                        if (navLinks[index]) {
                            navLinks[index].classList.add('bg-primary-600', 'text-white');
                            navLinks[index].classList.remove('bg-primary-50', 'text-primary-700');
                        }
                    }
                }
            });
        }

        // Update active nav on scroll (mobile only)
        if (window.innerWidth <= 768) {
            window.addEventListener('scroll', updateActiveNavItem);
        }
    