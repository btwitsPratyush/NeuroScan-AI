import { useRef, useEffect } from 'react';
import * as THREE from 'three';

export default function BrainAnimation() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(
      75,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000
    );

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true
    });
    renderer.setSize(
      containerRef.current.clientWidth,
      containerRef.current.clientHeight
    );
    renderer.setClearColor(0x000000, 0);
    containerRef.current.appendChild(renderer.domElement);

    const brainGroup = new THREE.Group();

    const hemisphereGeometry = new THREE.SphereGeometry(1.5, 32, 32, 0, Math.PI);
    const brainMaterial = new THREE.MeshPhongMaterial({
      color: 0xff6b9d,
      shininess: 30,
      emissive: 0x4a0e4e,
      emissiveIntensity: 0.2
    });

    const leftHemisphere = new THREE.Mesh(hemisphereGeometry, brainMaterial);
    leftHemisphere.rotation.y = Math.PI / 2;
    leftHemisphere.position.x = -0.1;
    brainGroup.add(leftHemisphere);

    const rightHemisphere = new THREE.Mesh(hemisphereGeometry, brainMaterial.clone());
    rightHemisphere.rotation.y = -Math.PI / 2;
    rightHemisphere.position.x = 0.1;
    brainGroup.add(rightHemisphere);

    for (let i = 0; i < 20; i++) {
      const gyrusGeometry = new THREE.TorusGeometry(
        0.3 + Math.random() * 0.5,
        0.05,
        8,
        16
      );
      const gyrusMaterial = new THREE.MeshPhongMaterial({
        color: 0xff6b9d,
        shininess: 30,
        emissive: 0x4a0e4e,
        emissiveIntensity: 0.2
      });
      const gyrus = new THREE.Mesh(gyrusGeometry, gyrusMaterial);
      gyrus.position.set(
        (Math.random() - 0.5) * 1.5,
        (Math.random() - 0.5) * 1.5,
        (Math.random() - 0.5) * 1.5
      );
      gyrus.rotation.set(
        Math.random() * Math.PI,
        Math.random() * Math.PI,
        Math.random() * Math.PI
      );
      brainGroup.add(gyrus);
    }

    scene.add(brainGroup);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);

    const frontLight = new THREE.DirectionalLight(0x4dabf7, 1);
    frontLight.position.set(0, 5, 5);
    scene.add(frontLight);

    const backLight = new THREE.DirectionalLight(0x845ef7, 0.8);
    backLight.position.set(0, -5, -5);
    scene.add(backLight);

    const pointLight1 = new THREE.PointLight(0x4dabf7, 1, 100);
    pointLight1.position.set(5, 5, 5);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff6b9d, 1, 100);
    pointLight2.position.set(-5, -5, 5);
    scene.add(pointLight2);

    camera.position.z = 5;

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      brainGroup.rotation.y += 0.005;
      brainGroup.rotation.x = Math.sin(Date.now() * 0.0005) * 0.1;
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(
        containerRef.current.clientWidth,
        containerRef.current.clientHeight
      );
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
        containerRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="w-full h-full min-h-[400px] md:min-h-[500px]"
    />
  );
}
