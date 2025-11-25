import { motion } from 'framer-motion';
import { Upload, Activity, FileSearch, Download, CheckCircle } from 'lucide-react';

export default function Detection() {
    const steps = [
        {
            icon: Upload,
            title: 'Upload MRI Scan',
            description: 'Securely upload patient MRI images in DICOM, PNG, or JPEG format',
            color: 'from-blue-500 to-cyan-500'
        },
        {
            icon: Activity,
            title: 'AI Processing',
            description: 'Our neural network analyzes the scan using advanced deep learning algorithms',
            color: 'from-cyan-500 to-teal-500'
        },
        {
            icon: FileSearch,
            title: 'Result Analysis',
            description: 'Receive detailed classification with confidence scores and visual heatmaps',
            color: 'from-teal-500 to-green-500'
        },
        {
            icon: Download,
            title: 'Export Report',
            description: 'Download comprehensive diagnostic reports for clinical documentation',
            color: 'from-green-500 to-emerald-500'
        }
    ];

    const tumorTypes = [
        {
            name: 'Glioma',
            description: 'Aggressive brain tumors arising from glial cells',
            //   prevalence: '35%',
            color: 'bg-red-500'
        },
        {
            name: 'Meningioma',
            description: 'Typically benign tumors of the meninges',
            //   prevalence: '30%',
            color: 'bg-orange-500'
        },
        {
            name: 'Pituitary',
            description: 'Tumors in the pituitary gland affecting hormones',
            //   prevalence: '15%',
            color: 'bg-purple-500'
        },
        {
            name: 'No Tumor',
            description: 'Healthy brain tissue with no abnormalities',
            //   prevalence: '20%',
            color: 'bg-green-500'
        }
    ];

    return (
        <section id="detection" className="py-20 bg-white">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-4xl font-bold text-gray-900 mb-4">
                        How Detection Works
                    </h2>
                    <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                        Our streamlined four-step process delivers accurate results in seconds
                    </p>
                </motion.div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-20">
                    {steps.map((step, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, y: 30 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: index * 0.1 }}
                            whileHover={{ y: -5, scale: 1.02 }}
                            className="relative group"
                        >
                            {index < steps.length - 1 && (
                                <div className="hidden lg:block absolute top-12 left-[60%] w-full h-0.5 bg-gradient-to-r from-blue-200 to-cyan-200" />
                            )}

                            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
                            <div className="relative bg-white rounded-xl p-6 shadow-lg border border-blue-100 group-hover:border-blue-300 transition-all duration-300">
                                <div className={`bg-gradient-to-br ${step.color} w-16 h-16 rounded-xl flex items-center justify-center mb-4 mx-auto`}>
                                    <step.icon className="w-8 h-8 text-white" />
                                </div>

                                <div className="absolute -top-3 -right-3 bg-blue-500 text-white w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shadow-lg">
                                    {index + 1}
                                </div>

                                <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">
                                    {step.title}
                                </h3>
                                <p className="text-gray-600 text-center text-sm">
                                    {step.description}
                                </p>
                            </div>
                        </motion.div>
                    ))}
                </div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-2xl p-8 md:p-12"
                >
                    <h3 className="text-3xl font-bold text-gray-900 mb-8 text-center">
                        Detectable Tumor Classifications
                    </h3>

                    <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {tumorTypes.map((tumor, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, scale: 0.9 }}
                                whileInView={{ opacity: 1, scale: 1 }}
                                viewport={{ once: true }}
                                transition={{ delay: index * 0.1 }}
                                whileHover={{ y: -5, scale: 1.02 }}
                                className="relative group"
                            >
                                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
                                <div className="relative bg-white rounded-xl p-6 shadow-lg border border-blue-100 group-hover:border-blue-300 transition-all duration-300 h-full">
                                    <div className="flex items-center justify-between mb-4">
                                        <div className={`${tumor.color} w-12 h-12 rounded-lg flex items-center justify-center shadow-md`}>
                                            <CheckCircle className="w-6 h-6 text-white" />
                                        </div>
                                        <div className="text-2xl font-bold text-gray-900">
                                            {tumor.prevalence}
                                        </div>
                                    </div>

                                    <h4 className="text-xl font-semibold text-gray-900 mb-2">
                                        {tumor.name}
                                    </h4>
                                    <p className="text-gray-600 text-sm">
                                        {tumor.description}
                                    </p>
                                </div>
                            </motion.div>
                        ))}
                    </div>

                    <motion.div
                        className="mt-12 relative group"
                        whileHover={{ y: -5, scale: 1.01 }}
                    >
                        <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-xl blur opacity-25 group-hover:opacity-40 transition-opacity" />
                        <div className="relative bg-white rounded-xl p-6 shadow-lg border border-blue-100 group-hover:border-blue-300 transition-all">
                            <div className="flex items-start space-x-4">
                                <div className="bg-blue-100 p-3 rounded-lg">
                                    <Activity className="w-6 h-6 text-blue-600" />
                                </div>
                                <div>
                                    <h4 className="text-lg font-semibold text-gray-900 mb-2">
                                        Clinical Validation
                                    </h4>
                                    <p className="text-gray-600">
                                        Our AI model has been validated against a panel of board-certified neuroradiologists,
                                        demonstrating comparable or superior performance in tumor detection and classification.
                                        The system undergoes continuous evaluation and improvement through partnerships with
                                        leading medical institutions worldwide.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            </div>
        </section>
    );
}
