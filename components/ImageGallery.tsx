import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ZoomIn, FileText, Brain, Activity } from 'lucide-react';

interface MedicalImage {
  id: number;
  title: string;
  type: string;
  description: string;
  url: string;
  diagnosis: string;
}

const medicalImages: MedicalImage[] = [
  {
    id: 1,
    title: 'Glioma Tumor - T2 MRI',
    type: 'Glioma',
    description: 'High-grade glioma showing irregular margins and significant mass effect',
    url: 'https://cdn.pixabay.com/photo/2024/04/13/01/21/ai-generated-8692966_1280.jpg',
    diagnosis: 'Aggressive tumor requiring immediate intervention'
  },
  {
    id: 2,
    title: 'Meningioma - Contrast Enhanced',
    type: 'Meningioma',
    description: 'Well-circumscribed extra-axial mass with dural attachment',
    url: 'https://cdn.xingosoftware.com/elektor/images/fetch/dpr_1,w_450,h_450,c_fit/https%3A%2F%2Fwww.elektormagazine.com%2Fassets%2Fupload%2Fimages%2F42%2F20250106214452_8393e6b5-02a5-4725-89be-c693b06b717e.png',
    diagnosis: 'Benign tumor with clear surgical margins'
  },
  {
    id: 3,
    title: 'Pituitary Adenoma - Sagittal View',
    type: 'Pituitary',
    description: 'Microadenoma in the anterior pituitary gland',
    url: 'https://img.freepik.com/premium-photo/colorful-human-brain-generative-ai-design_351967-692.jpg',
    diagnosis: 'Hormonal dysfunction detected, surgical candidate'
  },
  {
    id: 4,
    title: 'Normal Brain Scan - Axial T1',
    type: 'No Tumor',
    description: 'Normal brain anatomy with no abnormal findings',
    url: 'https://img.freepik.com/premium-photo/ai-generated-illustration-about-human-brain-computer-left-right-human-brain_441362-5549.jpg',
    diagnosis: 'No pathological findings detected'
  },
  {
    id: 5,
    title: 'Glioblastoma Multiforme',
    type: 'Glioma',
    description: 'Grade IV glioma with necrotic core and rim enhancement',
    url: 'https://my.clevelandclinic.org/-/scassets/images/org/health/articles/glioblastoma',
    diagnosis: 'Highly aggressive tumor requiring multimodal therapy'
  },
  {
    id: 6,
    title: 'Meningioma - Coronal Section',
    type: 'Meningioma',
    description: 'Parasagittal meningioma compressing cerebral cortex',
    url: 'https://plus.unsplash.com/premium_photo-1722622870646-761e6543aa8d?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=2445',
    diagnosis: 'Surgical resection recommended'
  },
  {
    id: 7,
    title: 'Post-Treatment Follow-up',
    type: 'No Tumor',
    description: 'Status post tumor resection showing no recurrence',
    url: 'https://t4.ftcdn.net/jpg/05/62/11/61/360_F_562116144_lxZOlafYtRtv8BzmKTKGcNby0D37ZVTZ.jpg',
    diagnosis: 'Complete response to treatment'
  },
  {
    id: 8,
    title: 'Pituitary Macroadenoma',
    type: 'Pituitary',
    description: 'Large pituitary mass with suprasellar extension',
    url: 'https://images.unsplash.com/photo-1711409645921-ef3db0501f96?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=3132',
    diagnosis: 'Visual field defects present, urgent surgery advised'
  }
];

export default function ImageGallery() {
  const [selectedImage, setSelectedImage] = useState<MedicalImage | null>(null);

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'Glioma':
        return 'bg-red-100 text-red-700 border-red-200';
      case 'Meningioma':
        return 'bg-orange-100 text-orange-700 border-orange-200';
      case 'Pituitary':
        return 'bg-purple-100 text-purple-700 border-purple-200';
      default:
        return 'bg-green-100 text-green-700 border-green-200';
    }
  };

  return (
    <section id="gallery" className="relative py-20 overflow-hidden">
      {/* Animated Gradient Background */}
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-purple-50 via-white to-cyan-50">
        <div className="absolute top-20 right-1/3 w-96 h-96 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000" />
        <div className="absolute bottom-20 left-1/3 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000" />
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Image Gallery
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            High-definition MRI scans showcasing various tumor types and diagnostic findings analyzed by our AI system
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {medicalImages.map((image, index) => (
            <motion.div
              key={image.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -8, scale: 1.02 }}
              className="group relative cursor-pointer"
              onClick={() => setSelectedImage(image)}
            >
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
              <div className="relative bg-white rounded-2xl overflow-hidden shadow-lg border border-purple-100 group-hover:border-purple-300 transition-all">
                <div className="aspect-square overflow-hidden bg-gray-100">
                  <img
                    src={image.url}
                    alt={image.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                    <ZoomIn className="w-12 h-12 text-white" />
                  </div>
                </div>

                <div className="p-4">
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border mb-2 ${getTypeColor(image.type)}`}>
                    {image.type}
                  </div>
                  <h3 className="font-bold text-gray-900 mb-1 line-clamp-1">
                    {image.title}
                  </h3>
                  <p className="text-sm text-gray-600 line-clamp-2">
                    {image.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      <AnimatePresence>
        {selectedImage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm"
            onClick={() => setSelectedImage(null)}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="relative bg-white rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={() => setSelectedImage(null)}
                className="absolute top-4 right-4 z-10 bg-white rounded-full p-2 shadow-lg hover:bg-gray-100 transition-colors"
              >
                <X className="w-6 h-6 text-gray-700" />
              </button>

              <div className="grid md:grid-cols-2 gap-6 p-6">
                <div className="relative rounded-xl overflow-hidden bg-gray-100">
                  <img
                    src={selectedImage.url}
                    alt={selectedImage.title}
                    className="w-full h-full object-cover"
                  />
                </div>

                <div className="space-y-4">
                  <div>
                    <div className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium border mb-3 ${getTypeColor(selectedImage.type)}`}>
                      {selectedImage.type}
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">
                      {selectedImage.title}
                    </h3>
                  </div>

                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-100">
                    <div className="flex items-start space-x-2">
                      <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
                      <div>
                        <div className="text-sm font-semibold text-blue-900 mb-1">
                          Clinical Description
                        </div>
                        <p className="text-sm text-blue-800">
                          {selectedImage.description}
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-4 border border-blue-100">
                    <div className="text-sm font-semibold text-gray-900 mb-2">
                      AI Diagnosis
                    </div>
                    <p className="text-sm text-gray-700">
                      {selectedImage.diagnosis}
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-2">
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-xs text-gray-600 mb-1">Confidence</div>
                      <div className="text-lg font-bold text-blue-600">97.3%</div>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="text-xs text-gray-600 mb-1">Scan Quality</div>
                      <div className="text-lg font-bold text-green-600">Excellent</div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
