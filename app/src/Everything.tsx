import React, { useState, useEffect } from "react";
import { Upload, Brain, Zap, Activity, FileText } from "lucide-react";

type Results = {
  original_image: string;
  inference_time: number;
  ground_truth?: string;
  prediction: string;
};

const TumorDetectionApp = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [ipAddress, setIpAddress] = useState("127.0.0.1");
  const [results, setResults] = useState<Results | null>(null);
  const [isApiHealthy, setIsApiHealthy] = useState(false);
  const [error, setError] = useState<string | { message: string } | null>(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith(".zip")) {
      setSelectedFile(file);
      setError(null);
    } else {
      setError("Please select a valid .nii file");
    }
  };
  useEffect(() => {
    // check api health
    const healthCheck = async () => {
      try {
        const response = await fetch(`http://${ipAddress}:5000/api/health`);
        if (!response.ok) {
          throw new Error("API is not reachable");
        }
        setIsApiHealthy(true);
      } catch (err: any) {
        setIsApiHealthy(false);
        setError("Error connecting to API: " + err.message);
      }
    };
    healthCheck();
  }, [ipAddress]);

  const handleSubmit = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append("nii_file", selectedFile);

    try {
      const response = await fetch(`http://${ipAddress}:5000/api/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        console.error("Prediction failed:", response.statusText);
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setResults(data);
    } catch (err: any) {
      setError("Error processing file: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">MATEK AI</h1>
                <p className="text-sm text-gray-600">
                  Liver Tumor Detection on FPGA
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <div className="flex items-center space-x-1">
                <input
                  name="ip address"
                  type="text"
                  placeholder="Enter IP address"
                  defaultValue={ipAddress}
                  onChange={(event) => setIpAddress(event.target.value)}
                  className="border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <span>{!isApiHealthy ? "🔴" : "🟢"}</span>
                <Zap className="w-4 h-4" />
                <span>KRIA KV260</span>
              </div>
              <div className="flex items-center space-x-1">
                <Activity className="w-4 h-4" />
                <span>Vitis AI</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Project Info */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <FileText className="w-5 h-5 mr-2" />
            Project Overview
          </h2>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                <Brain className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="font-medium">DeepLabV3-ResNet50</h3>
              <p className="text-sm text-gray-600">
                Fine-tuned on LiTS dataset
              </p>
            </div>
            <div className="text-center">
              <div className="bg-green-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                <Zap className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="font-medium">FPGA Acceleration</h3>
              <p className="text-sm text-gray-600">
                INT8 quantization with Vitis AI
              </p>
            </div>
            <div className="text-center">
              <div className="bg-purple-100 w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-3">
                <Activity className="w-6 h-6 text-purple-600" />
              </div>
              <h3 className="font-medium">Real-time Inference</h3>
              <p className="text-sm text-gray-600">
                Optimized for medical imaging
              </p>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">Upload NIfTI File</h2>

            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
              <input
                type="file"
                accept=".zip"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center space-y-4"
              >
                <Upload className="w-12 h-12 text-gray-400" />
                <div>
                  <p className="text-lg font-medium text-gray-700">
                    Drop your .nii zip file here or click to browse
                  </p>
                  <p className="text-sm text-gray-500">
                    LiTS dataset format supported
                  </p>
                </div>
              </label>
            </div>

            {selectedFile && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm font-medium text-blue-900">
                  Selected: {selectedFile.name}
                </p>
                <p className="text-sm text-blue-700">
                  Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={!selectedFile || isProcessing}
              className="w-full mt-6 bg-blue-600 text-white py-3 px-4 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  <span>Processing on FPGA...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4" />
                  <span>Run Inference</span>
                </>
              )}
            </button>
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            <h2 className="text-xl font-semibold mb-4">Segmentation Results</h2>

            {!results && !isProcessing && (
              <div className="text-center py-12 text-gray-500">
                <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                <p>Upload a NIfTI file to see segmentation results</p>
              </div>
            )}

            {isProcessing && (
              <div className="text-center py-12">
                <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-600 border-t-transparent mx-auto mb-4"></div>
                <p className="text-gray-600">
                  Processing on KRIA KV260 FPGA...
                </p>
                <p className="text-sm text-gray-500">
                  Running DeepLabV3-ResNet50 inference
                </p>
              </div>
            )}

            {results && (
              <div className="space-y-6">
                {/* Original Image */}
                <div>
                  <h3 className="font-medium mb-2">
                    Original CT Scan (Axial Slice)
                  </h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <img
                      src={`data:image/png;base64,${results.original_image}`}
                      alt="Original CT scan"
                      className="w-full h-auto rounded"
                    />
                  </div>
                </div>

                {/* Ground Truth */}
                {results.ground_truth && (
                  <div>
                    <h3 className="font-medium mb-2">
                      Ground Truth Segmentation
                    </h3>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <img
                        src={`data:image/png;base64,${results.ground_truth}`}
                        alt="Ground truth segmentation"
                        className="w-full h-auto rounded"
                      />
                    </div>
                  </div>
                )}

                {/* Prediction */}
                <div>
                  <h3 className="font-medium mb-2">FPGA Prediction</h3>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <img
                      src={`data:image/png;base64,${results.prediction}`}
                      alt="FPGA prediction"
                      className="w-full h-auto rounded"
                    />
                  </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm text-blue-700">Inference Time</p>
                    <p className="text-lg font-semibold text-blue-900">
                      {results.inference_time}ms
                    </p>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <p className="text-sm text-green-700">Detected Classes</p>
                    <p className="text-lg font-semibold text-green-900">
                      {results.unique_classes
                        ? results.unique_classes.join(", ")
                        : "N/A"}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TumorDetectionApp;
