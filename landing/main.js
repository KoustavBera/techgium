// Initialize Lenis for Smooth Scrolling
const lenis = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    direction: 'vertical',
    gestureDirection: 'vertical',
    smooth: true,
    mouseMultiplier: 1,
    smoothTouch: false,
    touchMultiplier: 2,
    infinite: false,
});

function raf(time) {
    lenis.raf(time);
    requestAnimationFrame(raf);
}
requestAnimationFrame(raf);

// Integrate Lenis with GSAP ScrollTrigger
gsap.registerPlugin(ScrollTrigger);

lenis.on('scroll', ScrollTrigger.update);
gsap.ticker.add((time)=>{
  lenis.raf(time * 1000);
});
gsap.ticker.lagSmoothing(0, 0);

// Populate Images using actual paths created by generate_image
document.getElementById('heroImg').src = 'file:///C:/Users/Swetanjana%20Maity/.gemini/antigravity/brain/4a96c56f-021b-4e53-8690-0bb95922abab/hero_abstract_1773897666048.png';
document.getElementById('sensorImg1').src = 'file:///C:/Users/Swetanjana%20Maity/.gemini/antigravity/brain/4a96c56f-021b-4e53-8690-0bb95922abab/sensor_visual_1773897682866.png';
document.getElementById('sensorImg2').src = 'file:///C:/Users/Swetanjana%20Maity/.gemini/antigravity/brain/4a96c56f-021b-4e53-8690-0bb95922abab/ai_health_1773897699352.png';
document.getElementById('sensorImg3').src = 'file:///C:/Users/Swetanjana%20Maity/.gemini/antigravity/brain/4a96c56f-021b-4e53-8690-0bb95922abab/hero_abstract_1773897666048.png';


// Custom Cursor Logic with motion
const cursor = document.querySelector('.cursor');
const hoverElements = document.querySelectorAll('a, button, .module-card');

document.addEventListener('mousemove', (e) => {
    // Basic cursor tracking
    gsap.to(cursor, {
        x: e.clientX,
        y: e.clientY,
        duration: 0.1,
        ease: "power2.out"
    });
});

hoverElements.forEach(el => {
    el.addEventListener('mouseenter', () => {
        cursor.classList.add('hovered');
    });
    el.addEventListener('mouseleave', () => {
        cursor.classList.remove('hovered');
    });
});

// Preloader Animation
window.addEventListener('load', () => {
    const tl = gsap.timeline();
    
    tl.to('.preloader-progress', {
        width: '100%',
        duration: 1.5,
        ease: 'power3.inOut'
    })
    .to('.preloader', {
        yPercent: -100,
        duration: 0.8,
        ease: 'power4.inOut'
    }, '+=0.2')
    // Hero Entrance
    .from('.hero-title', {
        y: 100,
        opacity: 0,
        duration: 1,
        ease: 'power3.out'
    }, '-=0.4')
    .from('.hero-subtitle', {
        y: 50,
        opacity: 0,
        duration: 1,
        ease: 'power3.out'
    }, '-=0.8')
    .from('.stat-card', {
        y: 50,
        opacity: 0,
        stagger: 0.1,
        duration: 0.8,
        ease: 'back.out(1.7)'
    }, '-=0.8')
    .from('.hero-image-wrapper', {
        scale: 0.8,
        opacity: 0,
        rotationY: -20,
        duration: 1.5,
        ease: 'power4.out'
    }, '-=1');
});

// Marquee Animation
gsap.to('.marquee-content', {
    xPercent: -100,
    ease: "none",
    duration: 20,
    repeat: -1,
});

// Horizontal Scroll for Modalities
const scrollContainer = document.querySelector('.horizontal-scroll-content');
const scrollOffset = scrollContainer.scrollWidth - window.innerWidth + 100; // 100 for padding

gsap.to(scrollContainer, {
    x: -scrollOffset,
    ease: "none",
    scrollTrigger: {
        trigger: ".modalities",
        pin: true,
        scrub: 1,
        end: () => "+=" + scrollOffset
    }
});

// Modules Entrance Animation
gsap.from('.module-card', {
    y: 100,
    opacity: 0,
    duration: 1,
    stagger: 0.2,
    ease: "power3.out",
    scrollTrigger: {
        trigger: ".modules",
        start: "top 70%",
    }
});

// Parallax effect on hero image
gsap.to('.hero-img', {
    yPercent: 20,
    ease: "none",
    scrollTrigger: {
        trigger: ".hero",
        start: "top top",
        end: "bottom top",
        scrub: true
    }
});

// Motion interaction using popmotion on module cards
const { spring } = window.popmotion;

document.querySelectorAll('.module-card').forEach(card => {
    // Optional: Using popmotion for complex spring physics on hover if GSAP isn't enough
    // For now GSAP basic hover in CSS handles the scale, but we can do a magnetic effect
    
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;
        
        gsap.to(card, {
            x: x * 0.05,
            y: y * 0.05,
            rotationY: x * 0.02,
            rotationX: -y * 0.02,
            duration: 0.4,
            ease: "power2.out",
            transformPerspective: 1000
        });
    });
    
    card.addEventListener('mouseleave', () => {
        gsap.to(card, {
            x: 0,
            y: 0,
            rotationY: 0,
            rotationX: 0,
            duration: 0.8,
            ease: "elastic.out(1, 0.3)"
        });
    });
});
