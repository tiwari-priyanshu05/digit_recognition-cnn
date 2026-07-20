import React, { useEffect, useRef } from 'react';

const Background3D = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    let animationFrameId;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    // Particle class for 3D simulation
    class Particle {
      constructor() {
        this.x = (Math.random() - 0.5) * 2000;
        this.y = (Math.random() - 0.5) * 2000;
        this.z = Math.random() * 2000;
        this.baseSize = Math.random() * 2 + 1;
        this.color = Math.random() > 0.5 ? '#a855f7' : '#06b6d4'; // Purple or Cyan
      }

      update(mouseX, mouseY, speed) {
        // Rotate around Y-axis
        const cosY = Math.cos(0.0005);
        const sinY = Math.sin(0.0005);
        const x1 = this.x * cosY - this.z * sinY;
        const z1 = this.z * cosY + this.x * sinY;
        this.x = x1;
        this.z = z1;

        // Rotate around X-axis slightly
        const cosX = Math.cos(0.0002);
        const sinX = Math.sin(0.0002);
        const y2 = this.y * cosX - this.z * sinX;
        const z2 = this.z * cosX + this.y * sinX;
        this.y = y2;
        this.z = z2;

        // Interactive mouse push
        const dx = mouseX - (width / 2);
        const dy = mouseY - (height / 2);
        this.x += (dx * 0.01 - this.x * 0.001) * 0.05;
        this.y += (dy * 0.01 - this.y * 0.001) * 0.05;

        // Wrap Z axis
        this.z -= speed;
        if (this.z <= 0) {
          this.z = 2000;
          this.x = (Math.random() - 0.5) * 2000;
          this.y = (Math.random() - 0.5) * 2000;
        }
      }

      draw(ctx, width, height) {
        // 3D projection
        const fov = 400; // Field of View
        const scale = fov / (fov + this.z);
        const projX = this.x * scale + width / 2;
        const projY = this.y * scale + height / 2;
        const size = this.baseSize * scale * 2;

        if (projX >= 0 && projX <= width && projY >= 0 && projY <= height) {
          ctx.beginPath();
          ctx.arc(projX, projY, size, 0, Math.PI * 2);
          ctx.fillStyle = this.color;
          // Alpha based on depth
          const alpha = Math.max(0, 1 - this.z / 2000);
          ctx.globalAlpha = alpha * 0.6;
          ctx.fill();
        }
      }
    }

    // Initialize particles
    const particleCount = 120;
    const particles = [];
    for (let i = 0; i < particleCount; i++) {
      particles.push(new Particle());
    }

    let mouseX = width / 2;
    let mouseY = height / 2;

    const handleMouseMove = (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('resize', handleResize);

    const animate = () => {
      ctx.fillStyle = '#09070f';
      ctx.globalAlpha = 1.0;
      ctx.fillRect(0, 0, width, height);

      // Draw background network lines first
      const fov = 400;
      for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        p1.update(mouseX, mouseY, 1.5);
        p1.draw(ctx, width, height);

        // Draw connections
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const dz = p1.z - p2.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

          if (dist < 250) {
            const scale1 = fov / (fov + p1.z);
            const scale2 = fov / (fov + p2.z);
            const x1 = p1.x * scale1 + width / 2;
            const y1 = p1.y * scale1 + height / 2;
            const x2 = p2.x * scale2 + width / 2;
            const y2 = p2.y * scale2 + height / 2;

            if (
              x1 >= 0 && x1 <= width && y1 >= 0 && y1 <= height &&
              x2 >= 0 && x2 <= width && y2 >= 0 && y2 <= height
            ) {
              ctx.beginPath();
              ctx.moveTo(x1, y1);
              ctx.lineTo(x2, y2);
              const avgZ = (p1.z + p2.z) / 2;
              const alpha = Math.max(0, (1 - avgZ / 2000) * (1 - dist / 250));
              ctx.globalAlpha = alpha * 0.15;
              ctx.strokeStyle = p1.color;
              ctx.lineWidth = 0.5 * scale1;
              ctx.stroke();
            }
          }
        }
      }

      animationFrameId = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed top-0 left-0 -z-10 w-full h-full block" />;
};

export default Background3D;
