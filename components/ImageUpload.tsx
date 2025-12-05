import React, { useState } from 'react';
import { UploadCloud } from 'lucide-react';

const ImageUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const validateFile = (uploadedFile: File) => {
    const name = uploadedFile.name.toLowerCase();
    const isImg = uploadedFile.type.startsWith('image/');
    const isMat = name.endsWith('.mat');

    if (isImg || isMat) return true;
    return false;
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    const uploadedFile = event.dataTransfer.files[0];
    if (uploadedFile && validateFile(uploadedFile)) {
      setFile(uploadedFile);
    } else {
      alert('Please upload PNG, JPG, JPEG, or MAT file.');
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile && validateFile(uploadedFile)) {
      setFile(uploadedFile);
    } else {
      alert('Please upload PNG, JPG, JPEG, or MAT file.');
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const json = await res.json();
      setResult(json);
    } catch (err: any) {
      setResult({ error: err.message });
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center">

      {/* Drag-drop box */}
      <div
        className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl transition-colors duration-200
          ${dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50 hover:border-gray-400'}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <UploadCloud className="w-12 h-12 text-gray-400 mb-4" />
        <p className="text-lg text-gray-600 mb-2">Drag and drop your MRI scan here, or</p>
        <label htmlFor="file-upload" className="cursor-pointer bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors duration-200">
          Click to select
        </label>
        <input
          id="file-upload"
          type="file"
          accept=".mat,image/png,image/jpeg,image/jpg"
          className="hidden"
          onChange={handleFileSelect}
        />

        {file && (
          <div className="mt-4 text-sm text-gray-700">
            Selected file: <span className="font-medium">{file.name}</span>
          </div>
        )}
        <p className="text-sm text-gray-500 mt-4">Supported: PNG, JPG, JPEG, MAT</p>
      </div>

      {/* Analyze button */}
      <button
        className="mt-4 bg-green-600 text-white px-5 py-2 rounded-md hover:bg-green-700"
        onClick={handleAnalyze}
        disabled={loading}
      >
        {loading ? "Analyzing..." : "Analyze Image"}
      </button>

      {/* Response */}
      {result && (
        <div className="mt-6 p-4 border rounded-md w-full max-w-lg bg-gray-50">
          <pre className="text-sm">{JSON.stringify(result, null, 2)}</pre>

          {/* Heatmap image if present */}
          {result.heatmap && (
            <img
              src={`data:image/png;base64,${result.heatmap}`}
              alt="heatmap"
              className="mt-4 rounded-lg shadow-md"
            />
          )}
        </div>
      )}
    </div>
  );
};

export default ImageUpload;