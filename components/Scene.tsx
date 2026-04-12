import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function Scene() {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

        renderer.setSize(window.innerWidth, window.innerHeight);
        containerRef.current.appendChild(renderer.domElement);

        // Particles
        const particlesGeometry = new THREE.BufferGeometry();
        const particlesCount = 700;

        const posArray = new Float32Array(particlesCount * 3);

        for (let i = 0; i < particlesCount * 3; i++) {
            posArray[i] = (Math.random() - 0.5) * 15;
        }

        particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

        // Material
        const material = new THREE.PointsMaterial({
            size: 0.02,
            color: 0x22d3ee, // Cyan
            transparent: true,
            opacity: 0.8,
        });

        // Mesh
        const particlesMesh = new THREE.Points(particlesGeometry, material);
        scene.add(particlesMesh);

        // Connecting lines
        const linesGeometry = new THREE.BufferGeometry();
        const linesMaterial = new THREE.LineBasicMaterial({
            color: 0x22d3ee,
            transparent: true,
            opacity: 0.15
        });

        // Calculate connections
        const linesPositions: number[] = [];
        const connectDistance = 1.5;

        for (let i = 0; i < particlesCount; i++) {
            for (let j = i + 1; j < particlesCount; j++) {
                const x1 = posArray[i * 3];
                const y1 = posArray[i * 3 + 1];
                const z1 = posArray[i * 3 + 2];

                const x2 = posArray[j * 3];
                const y2 = posArray[j * 3 + 1];
                const z2 = posArray[j * 3 + 2];

                const dist = Math.sqrt(Math.pow(x1 - x2, 2) + Math.pow(y1 - y2, 2) + Math.pow(z1 - z2, 2));

                if (dist < connectDistance) {
                    linesPositions.push(x1, y1, z1);
                    linesPositions.push(x2, y2, z2);
                }
            }
        }

        linesGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linesPositions, 3));
        const linesMesh = new THREE.LineSegments(linesGeometry, linesMaterial);
        scene.add(linesMesh);

        camera.position.z = 3;

        // Mouse interaction
        let mouseX = 0;
        let mouseY = 0;

        const handleMouseMove = (event: MouseEvent) => {
            mouseX = event.clientX / window.innerWidth - 0.5;
            mouseY = event.clientY / window.innerHeight - 0.5;
        };

        window.addEventListener('mousemove', handleMouseMove);

        // Animation
        const animate = () => {
            requestAnimationFrame(animate);

            particlesMesh.rotation.y += 0.001;
            particlesMesh.rotation.x += 0.001;

            linesMesh.rotation.y += 0.001;
            linesMesh.rotation.x += 0.001;

            // Gentle mouse follow
            particlesMesh.rotation.y += mouseX * 0.05;
            particlesMesh.rotation.x += mouseY * 0.05;

            linesMesh.rotation.y += mouseX * 0.05;
            linesMesh.rotation.x += mouseY * 0.05;

            renderer.render(scene, camera);
        };

        animate();

        // Resize handler
        const handleResize = () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
            if (containerRef.current) {
                containerRef.current.removeChild(renderer.domElement);
            }
            linesGeometry.dispose();
            material.dispose();
            linesMaterial.dispose();
            renderer.dispose();
        };
    }, []);

    return (
        <div
            ref={containerRef}
            className="fixed inset-0 -z-10 pointer-events-none bg-slate-900"
            style={{
                background: 'radial-gradient(circle at center, #1e293b 0%, #0f172a 100%)'
            }}
        />
    );
}
