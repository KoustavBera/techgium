/* ============================================================
   CHIRANJEEVI — main.js
   Awwwards-style: GSAP horizontal scroll pin on hero
   ============================================================ */

// ── Globals (Loaded via CDN in index.html) ────────────────
// gsap, ScrollTrigger, Lenis are available globally.

gsap.registerPlugin(ScrollTrigger);

// ── Preloader ──────────────────────────────────────────────
const preloader = document.querySelector('.preloader');
const preloaderProgress = document.querySelector('.preloader-progress');

let progress = 0;
const loadInterval = setInterval(() => {
    progress += Math.random() * 18;
    if (progress >= 100) {
        progress = 100;
        clearInterval(loadInterval);
        setTimeout(hidePreloader, 300);
    }
    preloaderProgress.style.width = progress + '%';
}, 80);

function hidePreloader() {
    gsap.to(preloader, {
        opacity: 0,
        duration: 0.6,
        ease: 'power2.inOut',
        onComplete: () => {
            preloader.style.display = 'none';
            initAll();
        }
    });
}

// ── Main init (after preloader) ────────────────────────────
function initAll() {
    initLenis();
    initCursor();
    initHeroHorizontalScroll();
    initMarquee();
    initCarousel();
    initEntranceAnims();
}

// ── Custom Cursor ──────────────────────────────────────────
function initCursor() {
    const cursor = document.querySelector('.cursor');
    if (!cursor) return;

    // Move cursor
    document.addEventListener('mousemove', (e) => {
        cursor.style.left = e.clientX + 'px';
        cursor.style.top = e.clientY + 'px';
    });

    // Add hovered class on interactive elements
    const interactables = document.querySelectorAll('a, button, .hc-card, .hc-card-video iframe');
    interactables.forEach(el => {
        el.addEventListener('mouseenter', () => cursor.classList.add('hovered'));
        el.addEventListener('mouseleave', () => cursor.classList.remove('hovered'));
    });
}

// ── Lenis Smooth Scroll ────────────────────────────────────
let lenis;
function initLenis() {
    lenis = new Lenis({
        duration: 1.2,
        easing: t => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
        smoothWheel: true,
    });

    // Connect Lenis to GSAP ticker
    gsap.ticker.add(time => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);

    // Also feed Lenis scroll position into ScrollTrigger
    lenis.on('scroll', ScrollTrigger.update);
}

// ── Hero: GSAP Horizontal Scroll Pin ──────────────────────
function initHeroHorizontalScroll() {
    const hero      = document.getElementById('hero');
    const track     = document.getElementById('heroHTrack');
    const progress  = document.getElementById('hcProgressFill');
    const carousel  = document.querySelector('.hero-carousel-outer');

    if (!hero || !track || !carousel) return;

    // We need the carousel strip width (everything to the right of the text panel)
    // The text panel is 50vw, the carousel adds extra width
    // Total horizontal travel = track.scrollWidth - window.innerWidth
    const getScrollAmount = () => -(track.scrollWidth - window.innerWidth);

    const st = gsap.to(track, {
        x: getScrollAmount,
        ease: 'none',
        scrollTrigger: {
            trigger: hero,
            start: 'top top',
            end: () => '+=' + Math.abs(getScrollAmount()),
            pin: true,
            anticipatePin: 1,
            scrub: 1.2,
            invalidateOnRefresh: true,
            onUpdate: (self) => {
                // Update the bottom progress bar
                if (progress) {
                    progress.style.width = (self.progress * 100) + '%';
                }
            }
        }
    });

    // Recalc on resize
    window.addEventListener('resize', () => ScrollTrigger.refresh());
}

// ── Marquee ────────────────────────────────────────────────
function initMarquee() {
    const marquee = document.getElementById('marquee1');
    if (!marquee) return;

    let x = 0;
    const speed = 0.4;
    const content = marquee.querySelector('.marquee-content');
    if (!content) return;
    const w = content.offsetWidth;

    function tick() {
        x -= speed;
        if (Math.abs(x) >= w) x = 0;
        marquee.querySelectorAll('.marquee-content').forEach(el => {
            el.style.transform = `translateX(${x}px)`;
        });
        requestAnimationFrame(tick);
    }
    tick();
}

// ── Architecture Carousel Auto-Swipe ───────────────────────
function initCarousel() {
    const carousel = document.querySelector('.arch-carousel');
    if (!carousel) return;

    // Clone the first slide to create a seamless infinite loop illusion
    const slides = carousel.querySelectorAll('.arch-slide');
    if (slides.length > 0) {
        carousel.appendChild(slides[0].cloneNode(true));
    }

    let autoScroll;
    const scrollDelay = 3000;

    function startScroll() {
        autoScroll = setInterval(() => {
            const isAtEnd = Math.ceil(carousel.scrollLeft + carousel.clientWidth) >= carousel.scrollWidth - 5;
            
            if (isAtEnd) {
                // Instantly teleport back to real first slide (0px) silently
                carousel.scrollTo({ left: 0, behavior: 'instant' });
                // Wait a frame, then smoothly scroll to the second slide
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        carousel.scrollBy({ left: carousel.clientWidth, behavior: 'smooth' });
                    });
                });
            } else {
                // Normal smooth scroll to next slide
                carousel.scrollBy({ left: carousel.clientWidth, behavior: 'smooth' });
            }
        }, scrollDelay);
    }

    function stopScroll() {
        clearInterval(autoScroll);
    }

    startScroll();

    // Pause on hover or interaction
    carousel.addEventListener('mouseenter', stopScroll);
    carousel.addEventListener('mouseleave', startScroll);
    carousel.addEventListener('touchstart', stopScroll, {passive: true});
    carousel.addEventListener('touchend', startScroll, {passive: true});
}

// ── Entrance animations (sections below hero) ──────────────
function initEntranceAnims() {
    // Arch layers slide in
    gsap.utils.toArray('.arch-layer').forEach((layer, i) => {
        gsap.from(layer, {
            opacity: 0,
            x: -40,
            duration: 0.7,
            delay: i * 0.1,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: layer,
                start: 'top 85%',
            }
        });
    });

    // OC cards pop up
    gsap.utils.toArray('.oc-card').forEach((card, i) => {
        gsap.from(card, {
            opacity: 0,
            y: 40,
            duration: 0.6,
            delay: i * 0.08,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: card,
                start: 'top 88%',
            }
        });
    });

    // POC metrics count-up feel
    gsap.utils.toArray('.poc-metric').forEach((m, i) => {
        gsap.from(m, {
            opacity: 0,
            y: 24,
            duration: 0.5,
            delay: i * 0.1,
            ease: 'power2.out',
            scrollTrigger: {
                trigger: m,
                start: 'top 90%',
            }
        });
    });

    // Agent text features stagger
    gsap.utils.toArray('.a-feat').forEach((feat, i) => {
        gsap.from(feat, {
            opacity: 0,
            x: -30,
            duration: 0.6,
            delay: i * 0.15,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.agent-features',
                start: 'top 85%',
            }
        });
    });

    // Agent images composite
    gsap.from('.av-img-main', {
        opacity: 0,
        scale: 0.9,
        y: 40,
        duration: 0.8,
        ease: 'power3.out',
        scrollTrigger: {
            trigger: '.agent-visuals',
            start: 'top 80%',
        }
    });

    gsap.from('.av-img-sub', {
        opacity: 0,
        scale: 0.8,
        x: 40,
        y: 40,
        duration: 0.8,
        delay: 0.3,
        ease: 'power3.out',
        scrollTrigger: {
            trigger: '.agent-visuals',
            start: 'top 80%',
        }
    });

    // MVP Roadmap Image
    gsap.from('.mvp-container', {
        opacity: 0,
        y: 60,
        duration: 1,
        ease: 'power3.out',
        scrollTrigger: {
            trigger: '.mvp-section',
            start: 'top 85%',
        }
    });
}
