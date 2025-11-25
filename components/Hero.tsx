import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Sparkles, UploadCloud, Activity, Brain, FileText, Download, AlertCircle, CheckCircle } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

const apiBaseFromEnv =
  typeof import.meta !== 'undefined'
    ? (
      (import.meta as unknown as { env?: { VITE_API_URL?: string } }).env?.VITE_API_URL
    )
    : undefined;

const API_BASE_URL = apiBaseFromEnv || 'http://localhost:8000';

const LABEL_MAP: Record<string, string> = {
  glioma: 'Glioma',
  meningioma: 'Meningioma',
  pituitary: 'Pituitary Tumor',
  no_tumor: 'No Tumor Detected',
};

type DetectionResult = {
  label: string;
  probability: number;
  heatmap?: string | null;
  timestamp: string;
};

// --- Animated Text Variants ---
const sentenceVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      delay: 0.2,
      staggerChildren: 0.08,
    },
  },
};

const wordVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

// --- Detection Steps Component ---
function DetectionSteps({ currentStep }) {
  const steps = [
    { number: 1, title: 'Upload Image', icon: UploadCloud },
    { number: 2, title: 'Preprocessing', icon: Activity },
    { number: 3, title: 'AI Analysis', icon: Brain },
    { number: 4, title: 'Results', icon: FileText }
  ];

  return (
    <div className="flex justify-between items-center mb-8">
      {steps.map((step, index) => (
        <div key={step.number} className="flex items-center flex-1">
          <div className="flex flex-col items-center">
            <motion.div
              initial={{ scale: 0.8, opacity: 0.5 }}
              animate={{
                scale: currentStep >= step.number ? 1 : 0.8,
                opacity: currentStep >= step.number ? 1 : 0.5
              }}
              className={`w-14 h-14 rounded-full flex items-center justify-center ${currentStep >= step.number
                ? 'bg-gradient-to-br from-blue-500 to-cyan-500 text-white'
                : 'bg-gray-200 text-gray-400'
                }`}
            >
              <step.icon className="w-7 h-7" />
            </motion.div>
            <span className="mt-2 text-xs font-medium text-gray-700">{step.title}</span>
          </div>
          {index < steps.length - 1 && (
            <div className="flex-1 h-1 mx-2 bg-gray-200 rounded">
              <motion.div
                initial={{ width: '0%' }}
                animate={{ width: currentStep > step.number ? '100%' : '0%' }}
                transition={{ duration: 0.5 }}
                className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded"
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// --- Detection Interface Component ---
function DetectionInterface({ uploadedFile, onBack, onNewAnalysis }: { uploadedFile: File | null; onBack: () => void; onNewAnalysis?: () => void }) {
  const [currentStep, setCurrentStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [detectionResult, setDetectionResult] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const startDetection = async () => {
    if (!uploadedFile) {
      setError('Please upload a scan before starting detection.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setDetectionResult(null);
    setCurrentStep(2);

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`${API_BASE_URL.replace(/\/$/, '')}/predict`, {
        method: 'POST',
        body: formData,
      });

      setCurrentStep(3);

      if (!response.ok) {
        let errorMessage = 'Unable to analyze the scan.';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.error || errorMessage;
        } catch (e) {
          errorMessage = await response.text() || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();

      setDetectionResult({
        label: data.label,
        probability:
          typeof data.probability === 'number'
            ? data.probability
            : Number(data.probability) || 0,
        heatmap: data.heatmap ?? null,
        timestamp: new Date().toLocaleString(),
      });

      setCurrentStep(4);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error during analysis.');
      setCurrentStep(4);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadReport = async () => {
    if (!detectionResult) return;

    const displayLabel = LABEL_MAP[detectionResult.label] || detectionResult.label;
    const labelLower = detectionResult.label.toLowerCase();
    const hasTumor = labelLower !== 'no_tumor' && labelLower !== 'notumor';
    const confidence = (detectionResult.probability * 100).toFixed(2);
    const risk = hasTumor
      ? (Number(confidence) >= 90 ? 'High' : 'Moderate')
      : 'No Risk';

    // Create PDF
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    let yPos = margin;

    // Helper function to add colored header
    const addHeader = () => {
      // Compact professional header
      doc.setFillColor(30, 64, 175); // Darker blue for professional look
      doc.rect(0, 0, pageWidth, 35, 'F');

      // Add subtle border line
      doc.setDrawColor(20, 50, 150);
      doc.setLineWidth(0.5);
      doc.line(0, 35, pageWidth, 35);

      // White text for header - smaller and more compact
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(18);
      doc.setFont('helvetica', 'bold');
      doc.text('NeuroScan AI', pageWidth / 2, 18, { align: 'center' });

      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(230, 230, 230);
      doc.text('Brain Tumor Detection Report', pageWidth / 2, 28, { align: 'center' });

      // Reset text color
      doc.setTextColor(0, 0, 0);
      yPos = 42;
    };

    // Helper function to add section header
    const addSectionHeader = (text: string, y: number) => {
      // Compact section header
      doc.setFillColor(37, 99, 235); // Blue-600
      doc.roundedRect(margin, y, pageWidth - 2 * margin, 7, 2, 2, 'F');

      doc.setTextColor(255, 255, 255);
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.text(text, margin + 5, y + 5);
      doc.setTextColor(0, 0, 0);
      return y + 10;
    };

    // Add header
    addHeader();

    // Report Information Table
    yPos = addSectionHeader('Report Information', yPos);

    const reportData = [
      ['Analysis Date', detectionResult.timestamp],
      ['Report ID', `NS-${Date.now()}`],
      ['Model Version', 'v2.5'],
    ];

    autoTable(doc, {
      startY: yPos,
      head: [['Field', 'Value']],
      body: reportData,
      theme: 'striped',
      headStyles: {
        fillColor: [37, 99, 235],
        textColor: 255,
        fontStyle: 'bold',
        fontSize: 9,
        halign: 'left',
        cellPadding: 2
      },
      bodyStyles: {
        fontSize: 9,
        cellPadding: 2,
        halign: 'left'
      },
      styles: {
        fontSize: 9,
        cellPadding: 2,
        lineColor: [200, 200, 200],
        lineWidth: 0.1
      },
      margin: { left: margin, right: margin },
      alternateRowStyles: { fillColor: [250, 250, 250] },
    });

    yPos = (doc as any).lastAutoTable.finalY + 8;

    // Detection Results Section
    yPos = addSectionHeader('Detection Results', yPos);

    const resultColor: [number, number, number] = hasTumor ? [239, 68, 68] : [34, 197, 94]; // Red or Green
    const riskColor: [number, number, number] = risk === 'High' ? [239, 68, 68] : risk === 'Moderate' ? [249, 115, 22] : [34, 197, 94];

    const resultsData = [
      ['Status', displayLabel],
      ['Confidence Level', `${confidence}%`],
      ['Model Probability', detectionResult.probability.toFixed(4)],
      ['Risk Assessment', risk],
    ];

    autoTable(doc, {
      startY: yPos,
      head: [['Parameter', 'Value']],
      body: resultsData,
      theme: 'striped',
      headStyles: {
        fillColor: resultColor,
        textColor: 255,
        fontStyle: 'bold',
        fontSize: 9,
        halign: 'left',
        cellPadding: 2
      },
      bodyStyles: {
        fillColor: (hasTumor ? [254, 242, 242] : [240, 253, 244]) as [number, number, number],
        fontSize: 9,
        cellPadding: 2,
        halign: 'left'
      },
      styles: {
        fontSize: 9,
        cellPadding: 2,
        lineColor: [200, 200, 200],
        lineWidth: 0.1
      },
      margin: { left: margin, right: margin },
      alternateRowStyles: { fillColor: (hasTumor ? [255, 250, 250] : [250, 255, 250]) as [number, number, number] },
      didParseCell: (data: any) => {
        if (data.row.index === resultsData.length - 1 && data.column.index === 1) {
          data.cell.styles.textColor = riskColor;
          data.cell.styles.fontStyle = 'bold';
          data.cell.styles.fontSize = 10;
        }
        if (data.row.index === 0 && data.column.index === 1) {
          data.cell.styles.textColor = resultColor;
          data.cell.styles.fontStyle = 'bold';
          data.cell.styles.fontSize = 10;
        }
      },
    });

    yPos = (doc as any).lastAutoTable.finalY + 8;

    // Add heatmap if available - using async approach to get proper dimensions
    if (detectionResult.heatmap) {
      if (yPos > pageHeight - 100) {
        doc.addPage();
        yPos = margin;
      }

      yPos = addSectionHeader('Model Heatmap Visualization', yPos);

      const availableWidth = pageWidth - 2 * margin;
      // Limit heatmap to max 80mm height to keep it compact
      const maxHeatmapHeight = 70;
      const availableHeight = Math.min(pageHeight - yPos - 60, maxHeatmapHeight);

      // Load image to get actual dimensions for proper aspect ratio
      const img = new Image();
      img.src = `data:image/png;base64,${detectionResult.heatmap}`;

      // Function to add image with proper dimensions
      const addImageToPDF = (imgWidth: number, imgHeight: number) => {
        // Calculate dimensions maintaining aspect ratio, but limit size
        const imgAspectRatio = imgWidth / imgHeight;
        // Use smaller width to keep image compact
        const maxWidth = Math.min(availableWidth * 0.7, 120); // Max 120mm width
        let finalWidth = maxWidth;
        let finalHeight = maxWidth / imgAspectRatio;

        // If height exceeds available space, scale down
        if (finalHeight > availableHeight) {
          finalHeight = availableHeight;
          finalWidth = availableHeight * imgAspectRatio;
        }

        // Center the image
        const xPos = margin + (availableWidth - finalWidth) / 2;

        // Add border around image
        doc.setDrawColor(200, 200, 200);
        doc.setLineWidth(0.5);
        doc.roundedRect(xPos - 2, yPos - 2, finalWidth + 4, finalHeight + 4, 2, 2, 'S');

        // Add image with proper aspect ratio
        doc.addImage(
          `data:image/png;base64,${detectionResult.heatmap}`,
          'PNG',
          xPos,
          yPos,
          finalWidth,
          finalHeight
        );

        // Add compact caption
        const captionY = yPos + finalHeight + 4;
        doc.setFontSize(8);
        doc.setTextColor(100, 100, 100);
        doc.setFont('helvetica', 'italic');
        doc.text('Heatmap overlay showing model attention regions', pageWidth / 2, captionY, { align: 'center' });
        yPos = captionY + 5;
      };

      // Wait for image to load with promise
      await new Promise<void>((resolve) => {
        if (img.complete && img.width > 0 && img.height > 0) {
          // Image already loaded
          addImageToPDF(img.width, img.height);
          resolve();
        } else {
          img.onload = () => {
            if (img.width > 0 && img.height > 0) {
              addImageToPDF(img.width, img.height);
            }
            resolve();
          };

          img.onerror = () => {
            doc.setFontSize(9);
            doc.setTextColor(128, 128, 128);
            doc.text('Heatmap visualization unavailable', margin + 5, yPos);
            yPos += 10;
            resolve();
          };
        }
      });
    }

    // Recommendations Section - compact
    if (yPos > pageHeight - 50) {
      doc.addPage();
      yPos = margin;
    }

    yPos = addSectionHeader('Recommendations', yPos);

    const recommendations = hasTumor
      ? [
        'Consult a neurologist or oncologist for clinical correlation',
        'Further diagnostic imaging may be required',
        'This AI analysis must be confirmed by medical professionals',
      ]
      : [
        'Continue regular health monitoring',
        'Maintain a healthy lifestyle',
        'This AI analysis should be confirmed by medical professionals',
      ];

    doc.setFontSize(9);
    doc.setTextColor(51, 51, 51);
    recommendations.forEach((rec, index) => {
      if (yPos > pageHeight - 30) {
        doc.addPage();
        yPos = margin;
      }
      // Compact bullet point
      doc.setFillColor(59, 130, 246);
      doc.circle(margin + 3, yPos - 0.5, 1, 'F');
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(51, 51, 51);
      const lines = doc.splitTextToSize(rec, pageWidth - 2 * margin - 12);
      doc.text(lines, margin + 8, yPos, { maxWidth: pageWidth - 2 * margin - 12 });
      yPos += lines.length * 5 + 1;
    });

    yPos += 5;

    // Disclaimer Section - compact
    if (yPos > pageHeight - 40) {
      doc.addPage();
      yPos = margin;
    }

    // Compact disclaimer box
    const disclaimerHeight = 25;
    doc.setFillColor(255, 249, 219); // Lighter yellow background
    doc.setDrawColor(234, 179, 8); // Yellow border
    doc.setLineWidth(0.5);
    doc.roundedRect(margin, yPos, pageWidth - 2 * margin, disclaimerHeight, 2, 2, 'FD');

    // Compact warning
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(184, 78, 12); // Darker yellow/orange
    doc.text('⚠ IMPORTANT DISCLAIMER', pageWidth / 2, yPos + 7, { align: 'center' });

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(120, 53, 15);
    const disclaimerText = 'This report is generated by AI and should not be used as a sole basis for medical decisions. Always consult with qualified healthcare professionals for proper diagnosis and treatment.';
    const splitDisclaimer = doc.splitTextToSize(disclaimerText, pageWidth - 2 * margin - 15);
    doc.text(splitDisclaimer, pageWidth / 2, yPos + 15, { align: 'center', maxWidth: pageWidth - 2 * margin - 15 });

    yPos += disclaimerHeight + 5;

    // Professional Footer with divider
    doc.setDrawColor(200, 200, 200);
    doc.setLineWidth(0.3);
    doc.line(margin, pageHeight - 20, pageWidth - margin, pageHeight - 20);

    doc.setFontSize(8);
    doc.setTextColor(120, 120, 120);
    doc.setFont('helvetica', 'normal');
    doc.text(
      `Generated by NeuroScan AI v2.5 • ${new Date().toLocaleString()} • Confidential Medical Report`,
      pageWidth / 2,
      pageHeight - 12,
      { align: 'center' }
    );

    // Add page numbers if multiple pages
    const totalPages = (doc as any).internal.getNumberOfPages();
    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      doc.setFontSize(8);
      doc.setTextColor(150, 150, 150);
      doc.text(
        `Page ${i} of ${totalPages}`,
        pageWidth - margin - 10,
        pageHeight - 12,
        { align: 'right' }
      );
    }

    // Save PDF
    doc.save(`NeuroScan_Report_${Date.now()}.pdf`);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full h-full bg-white/80 backdrop-blur-md rounded-2xl shadow-xl p-6 overflow-y-auto"
    >
      <div className="mb-6 text-center">
        <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent mb-2">
          Brain Tumor Detection
        </h2>
        <p className="text-sm text-gray-600">AI-powered analysis in progress</p>
      </div>

      <DetectionSteps currentStep={currentStep} />

      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="text-sm font-semibold text-gray-700 mb-2">Uploaded Image</h3>
            {uploadedFile && (
              <img
                src={URL.createObjectURL(uploadedFile)}
                alt="MRI Scan"
                className="w-full h-48 object-cover rounded-lg border-2 border-gray-200"
              />
            )}
          </div>

          {currentStep >= 2 && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <h3 className="text-sm font-semibold text-gray-700 mb-2">
                {currentStep === 2 && 'Preprocessing...'}
                {currentStep === 3 && 'AI Analysis in Progress...'}
                {currentStep === 4 && 'Detection Complete'}
              </h3>

              {isAnalyzing && (
                <div className="bg-blue-50 rounded-lg p-4 h-48 flex flex-col items-center justify-center">
                  <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-3"></div>
                  <p className="text-gray-700 text-sm font-medium">Analyzing brain scan...</p>
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">
                  {error}
                </div>
              )}

              {detectionResult && (
                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4 space-y-3">
                  {(() => {
                    const displayLabel = LABEL_MAP[detectionResult.label] || detectionResult.label;
                    const labelLower = detectionResult.label.toLowerCase();
                    const hasTumor = labelLower !== 'no_tumor' && labelLower !== 'notumor';
                    const confidence = (detectionResult.probability * 100).toFixed(2);
                    const risk = hasTumor
                      ? (Number(confidence) >= 90 ? 'High' : 'Moderate')
                      : 'No Risk';

                    return (
                      <>
                        <div className={`flex items-center gap-2 ${hasTumor ? 'text-orange-600' : 'text-green-600'}`}>
                          <CheckCircle className="w-5 h-5" />
                          <span className="font-semibold text-sm">Analysis Complete</span>
                        </div>

                        <div className="space-y-2">
                          <div className={`rounded-lg p-3 ${hasTumor ? 'bg-orange-50' : 'bg-green-50'}`}>
                            <p className="text-xs text-gray-600">{hasTumor ? 'Detected Tumor' : 'Status'}</p>
                            <p className={`text-base font-bold ${hasTumor ? 'text-orange-600' : 'text-green-600'}`}>
                              {displayLabel}
                            </p>
                          </div>

                          <div className="grid grid-cols-2 gap-2">
                            <div className="bg-white rounded-lg p-2">
                              <p className="text-xs text-gray-600">Confidence</p>
                              <p className="text-sm font-semibold text-gray-800">{confidence}%</p>
                            </div>
                            <div className="bg-white rounded-lg p-2">
                              <p className="text-xs text-gray-600">Timestamp</p>
                              <p className="text-sm font-semibold text-gray-800">{detectionResult.timestamp}</p>
                            </div>
                            <div className="bg-white rounded-lg p-2 col-span-2">
                              <p className="text-xs text-gray-600">Risk Level</p>
                              <p
                                className={`text-sm font-semibold ${risk === 'High'
                                  ? 'text-red-600'
                                  : risk === 'Moderate'
                                    ? 'text-orange-600'
                                    : 'text-green-600'
                                  }`}
                              >
                                {risk}
                              </p>
                            </div>
                          </div>
                        </div>

                        {detectionResult.heatmap && (
                          <div>
                            <p className="text-xs text-gray-600 mb-2">Model Heatmap</p>
                            <img
                              src={`data:image/png;base64,${detectionResult.heatmap}`}
                              alt="Model heatmap overlay"
                              className="w-full rounded-lg border border-gray-200 shadow-sm"
                            />
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              )}
            </motion.div>
          )}
        </div>

        <div className="flex flex-wrap gap-3 justify-center pt-4 border-t border-gray-200">
          {currentStep === 1 && (
            <>
              <button
                onClick={startDetection}
                className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-6 py-2 rounded-lg hover:shadow-lg hover:scale-105 transition-all duration-300 font-semibold text-sm flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
                disabled={isAnalyzing}
              >
                <Brain className="w-4 h-4" />
                {isAnalyzing ? 'Preparing...' : 'Start Detection'}
              </button>
              <button
                onClick={onBack}
                className="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition-colors font-semibold text-sm"
              >
                Cancel
              </button>
            </>
          )}

          {currentStep === 4 && detectionResult && (
            <>
              <button
                onClick={downloadReport}
                className="bg-gradient-to-r from-green-500 to-emerald-500 text-white px-6 py-2 rounded-lg hover:shadow-lg hover:scale-105 transition-all duration-300 font-semibold text-sm flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Download Report
              </button>
              <button
                onClick={() => {
                  if (onNewAnalysis) {
                    onNewAnalysis();
                  } else {
                    onBack();
                  }
                }}
                className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-6 py-2 rounded-lg hover:shadow-lg hover:scale-105 transition-all duration-300 font-semibold text-sm"
              >
                New Analysis
              </button>
            </>
          )}
        </div>

        {detectionResult && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded"
          >
            <div className="flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-yellow-600 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-semibold text-yellow-800 text-xs">Medical Disclaimer</p>
                <p className="text-xs text-yellow-700 mt-1">
                  This AI analysis is for informational purposes only. Consult with a qualified healthcare provider.
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default function Hero({ onStartDetectionReady }: { onStartDetectionReady?: (handler: () => void) => void }) {
  const [showUpload, setShowUpload] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [showDetection, setShowDetection] = useState(false);

  const handleStartDetection = useCallback(() => {
    setUploadedFile(null);
    setShowUpload(true);
    setShowDetection(false);
  }, []);

  useEffect(() => {
    if (onStartDetectionReady) {
      onStartDetectionReady(handleStartDetection);
    }
  }, [onStartDetectionReady, handleStartDetection]);

  const handleFileUploaded = useCallback((file: File | null) => {
    setUploadedFile(file);
    if (file) {
      setShowDetection(true);
      setShowUpload(false);
    }
  }, []);

  const handleCancelUpload = () => {
    setUploadedFile(null);
    setShowUpload(false);
    setShowDetection(false);
  };

  const handleNewAnalysis = useCallback(() => {
    // Reset all state and open upload window
    setUploadedFile(null);
    setShowDetection(false);
    setShowUpload(true);
  }, []);

  const heroHeading = 'Advanced Brain Tumor Detection System';

  return (
    <section
      id="home"
      className="relative min-h-screen flex items-center pt-24 pb-12 overflow-hidden"
    >
      {/* --- ANIMATED GRADIENT BACKGROUND --- */}
      <div className="absolute inset-0 -z-10 bg-gradient-to-br from-blue-50 via-white to-cyan-50">
        {/* Animated gradient orbs */}
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob" />
        <div className="absolute top-40 right-10 w-72 h-72 bg-cyan-400 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000" />
      </div>

      {/* --- FLOATING BRAIN & NEURAL ICONS --- */}
      <div className="absolute inset-0 -z-5 overflow-hidden pointer-events-none">
        {/* Floating Brain Icons */}
        <motion.div
          animate={{
            y: [0, -30, 0],
            rotate: [0, 10, 0],
            scale: [1, 1.1, 1]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-1/4 left-10 opacity-10"
        >
          <Brain className="w-20 h-20 text-blue-600" />
        </motion.div>

        <motion.div
          animate={{
            y: [0, 25, 0],
            rotate: [0, -8, 0],
            scale: [1, 1.15, 1]
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
          className="absolute bottom-1/4 right-20 opacity-10"
        >
          <Activity className="w-24 h-24 text-cyan-600" />
        </motion.div>


        {/* Neural Network Lines */}
        <svg className="absolute inset-0 w-full h-full opacity-5" xmlns="http://www.w3.org/2000/svg">
          <motion.line
            x1="60%" y1="30%" x2="90%" y2="70%"
            stroke="#06b6d4"
            strokeWidth="2"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.3 }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          />
          <motion.line
            x1="30%" y1="80%" x2="70%" y2="40%"
            stroke="#bf1e66ff"
            strokeWidth="2"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.3 }}
            transition={{ duration: 5, repeat: Infinity, ease: "easeInOut", delay: 2 }}
          />
        </svg>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 relative z-10 w-full">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* --- LEFT CONTENT --- */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-500 to-cyan-500 text-white px-5 py-2.5 rounded-full text-sm font-semibold shadow-lg shadow-blue-500/30"
            >
              <Sparkles className="w-4 h-4 animate-pulse" />
              <span>AI-Powered Medical Imaging</span>
            </motion.div>

            {/* Animated Heading */}
            <motion.h1
              className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 leading-tight"
              variants={sentenceVariants}
              initial="hidden"
              animate="visible"
            >
              {heroHeading.split(' ').map((word, index) => (
                <motion.span
                  key={index}
                  variants={wordVariants}
                  className="inline-block"
                >
                  {word === 'Detection' ? (
                    <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                      Detection
                    </span>
                  ) : word === 'System' ? (
                    <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                      System
                    </span>
                  ) : (
                    word
                  )}{' '}
                </motion.span>
              ))}
            </motion.h1>

            <motion.p
              className="text-lg text-gray-600 leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 }}
            >
              NeuroScan AI leverages cutting-edge deep learning technology to
              provide rapid, accurate brain tumor detection from MRI scans.
            </motion.p>

            <motion.div
              className="flex flex-col sm:flex-row gap-4 pt-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1 }}
            >
              <motion.button
                whileHover={{ scale: 1.05, boxShadow: "0 20px 40px rgba(59, 130, 246, 0.4)" }}
                whileTap={{ scale: 0.95 }}
                onClick={handleStartDetection}
                className="relative flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-600 to-cyan-600 text-white px-8 py-4 rounded-xl shadow-xl shadow-blue-500/30 transition-all duration-300 font-bold text-lg group overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-blue-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <Brain className="w-6 h-6 relative z-10" />
                <span className="relative z-10">Start Detection</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform relative z-10" />
              </motion.button>
            </motion.div>

            {/* --- UPGRADED STATS (Glassmorphism) --- */}
            <motion.div
              className="grid grid-cols-3 gap-4 pt-8"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 1.2 }}
            >
              <StatCard value="98.5%" label="Accuracy" />
              <StatCard value="<1s" label="Processing" isCyan />
              <StatCard value="50k+" label="Scans Analyzed" />
            </motion.div>
          </div>

          {/* --- RIGHT VISUAL / UPLOAD ZONE --- */}
          <div className="relative w-full h-96 lg:h-[500px]">
            <AnimatePresence mode="wait">
              {showDetection && uploadedFile ? (
                <DetectionInterface
                  key="detection"
                  uploadedFile={uploadedFile}
                  onBack={handleCancelUpload}
                  onNewAnalysis={handleNewAnalysis}
                />
              ) : showUpload ? (
                <UploadDropzone
                  key="upload"
                  onCancel={handleCancelUpload}
                  onFileSelect={handleFileUploaded}
                />
              ) : (
                <ScanVisual key="visual" />
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </section>
  );
}

// --- Sub-component for the Stats ---
const StatCard = ({
  value,
  label,
  isCyan = false,
}: {
  value: string;
  label: string;
  isCyan?: boolean;
}) => (
  <motion.div
    whileHover={{ y: -5, scale: 1.05 }}
    className="relative group"
  >
    <div className="relative text-center bg-white p-6 rounded-2xl shadow-lg border border-blue-100 group-hover:border-blue-300 transition-all">
      <div className={`text-3xl font-bold mb-1 ${isCyan ? 'text-cyan-600' : 'text-blue-600'}`}>
        {value}
      </div>
      <div className="text-sm text-gray-600 font-medium">{label}</div>
    </div>
  </motion.div>
);

// --- Sub-component for the Scanning Visual ---
const ScanVisual = () => (
  <motion.div
    className="relative w-full h-full flex items-center justify-center"
    animate={{ translateY: ['-5px', '5px'] }}
    transition={{
      duration: 3,
      repeat: Infinity,
      repeatType: 'mirror',
      ease: 'easeInOut',
    }}
  >
    <div className="w-[450px] h-[450px] max-w-full max-h-full bg-white/70 rounded-full p-2 shadow-2xl backdrop-blur-sm border border-white/50">
      <img
        src="/result_0.png"
        alt="Brain Scan Visual"
        className="w-full h-full object-contain rounded-full"
      />
      <motion.div
        className="absolute top-0 left-0 right-0 h-1.5 bg-cyan-300/80"
        style={{
          boxShadow: '0 0 20px 5px rgba(6, 182, 212, 0.7)',
        }}
        animate={{ translateY: '450px' }}
        transition={{
          duration: 2.5,
          repeat: Infinity,
          repeatType: 'mirror',
          ease: 'easeInOut',
        }}
      />
    </div>
  </motion.div>
);

// --- Sub-component for the Upload Dropzone ---
const UploadDropzone = ({
  onCancel,
  onFileSelect,
}: {
  onCancel: () => void;
  onFileSelect: (file: File | null) => void;
}) => {
  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      onFileSelect(file);
    } else {
      onFileSelect(null);
    }
  };

  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-white/60 backdrop-blur-md rounded-2xl shadow-xl border-2 border-dashed border-blue-300 p-8">
      <input
        type="file"
        id="file-upload"
        className="hidden"
        accept=".jpg,.jpeg,.png,.dicom"
        onChange={handleFileChange}
      />

      <label
        htmlFor="file-upload"
        className="flex flex-col items-center justify-center space-y-4 cursor-pointer"
      >
        <div className="p-4 bg-blue-100 rounded-full">
          <UploadCloud className="w-12 h-12 text-blue-500" />
        </div>
        <h3 className="text-2xl font-semibold text-gray-800">
          Upload MRI Scan
        </h3>
        <p className="text-gray-600 text-center">
          Drag & drop your file here, or{' '}
          <span className="text-blue-600 font-medium">browse</span>
        </p>
        <p className="text-xs text-gray-500">Supports: .jpg, .png, .dicom</p>
      </label>

      <button
        onClick={onCancel}
        className="text-sm text-gray-600 hover:text-red-500 transition-colors pt-4"
      >
        Cancel
      </button>
    </div>
  );
};