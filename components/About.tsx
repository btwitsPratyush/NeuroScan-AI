import { motion } from 'framer-motion';
import { Brain, Cpu, Shield, Zap, Target, TrendingUp, Activity } from 'lucide-react';

export default function About() {
  const features = [
    {
      icon: Brain,
      title: 'Advanced Neural Networks',
      description: 'Utilizing state-of-the-art deep learning architectures trained on over 50,000 validated medical scans'
    },
    {
      icon: Zap,
      title: 'Real-Time Analysis',
      description: 'Process and analyze MRI scans in under 2 seconds with GPU-accelerated inference'
    },
    {
      icon: Target,
      title: '98.5% Accuracy',
      description: 'Industry-leading precision in detecting and classifying four major tumor types'
    },
    {
      icon: Shield,
      title: 'Secure',
      description: 'Enterprise-grade security with end-to-end encryption, strict access controls, regular security audits, data minimization, secure development lifecycle, and a comprehensive incident response plan.'

    },
    {
      icon: Cpu,
      title: 'Explainable AI',
      description: 'Grad-CAM heatmaps provide visual explanations highlighting areas of clinical significance'
    },
    {
      icon: TrendingUp,
      title: 'Continuous Learning',
      description: 'Model updates quarterly with the latest medical research and clinical data'
    }
  ];

  return (
    <section id="about" className="relative py-20 overflow-hidden">
      {/* Animated Gradient Background */}
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-cyan-50 via-white to-blue-50">
        <div className="absolute top-20 right-10 w-96 h-96 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob" />
        <div className="absolute bottom-20 left-10 w-96 h-96 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-gray-900 mb-0">
            About NeuroScan AI
          </h2>
          <motion.div
            className="h-1 bg-gradient-to-r from-blue-500 to-cyan-500 mx-auto rounded-full"
            initial={{ width: 0 }}
            whileInView={{ width: 350 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 items-center mb-16">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <h3 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              Revolutionizing Brain Tumor Detection Through AI
            </h3>

            <p className="text-lg text-gray-700 leading-relaxed">
              NeuroScan AI represents a breakthrough in medical imaging analysis, combining cutting-edge artificial intelligence with clinical expertise to deliver rapid, accurate brain tumor detection. Our proprietary deep learning model has been meticulously trained on a diverse dataset of over 50,000 validated MRI scans, encompassing multiple imaging protocols and patient demographics.
            </p>

            <p className="text-lg text-gray-700 leading-relaxed">
              The system employs advanced convolutional neural networks (CNNs) and attention mechanisms to identify and classify four primary tumor types: gliomas, meningiomas, pituitary tumors, and normal brain tissue. Each scan is processed through multiple neural pathways, analyzing texture patterns, morphological features, and intensity distributions to achieve an industry-leading accuracy rate of 98.5%.
            </p>

            <p className="text-lg text-gray-700 leading-relaxed">
              What sets NeuroScan AI apart is our commitment to explainable artificial intelligence. Every diagnosis is accompanied by Grad-CAM visualization technology, which highlights the specific regions of interest that influenced the model's decision. This transparency empowers radiologists and neurosurgeons to validate AI findings with their clinical judgment, fostering a collaborative diagnostic approach.
            </p>

            <p className="text-lg text-gray-700 leading-relaxed">
              Our platform supports all major MRI sequences including T1-weighted, T2-weighted, FLAIR, and contrast-enhanced imaging. The AI engine adapts to varying scan qualities and protocols, ensuring consistent performance across different imaging centers and equipment manufacturers. With sub-second processing times and HIPAA-compliant infrastructure, NeuroScan AI seamlessly integrates into existing clinical workflows, reducing diagnostic turnaround times while maintaining the highest standards of patient data security.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="relative group"
          >
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-3xl blur-xl opacity-30 group-hover:opacity-40 transition-opacity" />
            <div className="relative bg-white rounded-2xl p-8 shadow-2xl">
              <img
                src="https://cdn.britannica.com/36/152936-050-92C2B3BD/Magnetic-resonance-imaging-types-abnormalities.jpg"
                alt="Medical professional analyzing brain scans"
                className="rounded-xl w-full h-auto shadow-lg"
              />
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="absolute -bottom-6 -right-6 bg-gradient-to-br from-blue-600 to-cyan-600 text-white p-6 rounded-xl shadow-2xl"
              >
                <div className="text-3xl font-bold">50k+</div>
                <div className="text-sm opacity-90">Scans Analyzed</div>
              </motion.div>
            </div>
          </motion.div>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -5, scale: 1.02 }}
              className="relative group"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
              <div className="relative bg-white rounded-2xl p-6 shadow-lg border border-blue-100 group-hover:border-blue-300 transition-all h-full">
                <div className="bg-gradient-to-br from-blue-500 to-cyan-500 w-14 h-14 rounded-xl flex items-center justify-center mb-4 shadow-lg">
                  <feature.icon className="w-7 h-7 text-white" />
                </div>
                <h4 className="text-xl font-bold text-gray-900 mb-3">
                  {feature.title}
                </h4>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
