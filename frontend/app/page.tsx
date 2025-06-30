"use client"

import React, { useState, useCallback } from "react"
import { Upload, ImageIcon, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface PredictionResult {
  predictedClass: string
  confidence: number
}

const diseaseDetails: Record<
  string,
  { name: string; description: string; treatment: string }
> = {
  akiec: {
    name: "Actinic Keratoses",
    description:
      "Actinic Keratoses are rough, scaly patches from years of sun damage. They’re considered pre-cancerous and can develop into squamous cell carcinoma.",
    treatment:
      "Cryotherapy, topical creams like 5-FU or imiquimod, and photodynamic therapy are common treatments. Regular monitoring and sun protection are key.",
  },
  bcc: {
    name: "Basal Cell Carcinoma",
    description:
      "A common, slow-growing form of skin cancer. It usually appears as a pearly or waxy bump and rarely spreads but needs treatment to prevent local tissue damage.",
    treatment:
      "Mohs surgery, excision, cryotherapy, and topical medications. Prompt treatment leads to excellent outcomes.",
  },
  bkl: {
    name: "Benign Keratosis-like Lesions",
    description:
      "Includes non-cancerous growths like seborrheic keratosis. These can look similar to melanoma but are harmless.",
    treatment:
      "No treatment needed unless removal is desired for cosmetic reasons. Can be removed via cryotherapy or curettage.",
  },
  df: {
    name: "Dermatofibroma",
    description:
      "A firm, benign skin nodule, usually on the legs. Typically harmless, but may be itchy or sensitive.",
    treatment:
      "Often left untreated. If bothersome, excision or cryotherapy is an option.",
  },
  mel: {
    name: "Melanoma",
    description:
      "A serious form of skin cancer that can spread quickly. Often appears as an irregular dark mole. Early detection is vital.",
    treatment:
      "Surgical removal is standard. Advanced cases may require immunotherapy, chemotherapy, or targeted therapy.",
  },
  nv: {
    name: "Melanocytic Nevi (Moles)",
    description:
      "Common pigmented skin lesions formed by clusters of melanocytes. Usually harmless, but monitor for changes.",
    treatment:
      "No treatment needed unless changes occur. Surgical excision is done if malignancy is suspected.",
  },
  vasc: {
    name: "Vascular Lesions",
    description:
      "Includes angiomas and hemangiomas. They appear red or purple due to blood vessels and are usually harmless.",
    treatment:
      "Observation is typical. Laser or surgical options exist if removal is needed.",
  },
}

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
    setPrediction(null)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      const files = e.dataTransfer.files
      if (files.length > 0 && files[0].type.startsWith("image/")) {
        handleFileSelect(files[0])
      }
    },
    [handleFileSelect]
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelect(e.target.files[0])
      }
    },
    [handleFileSelect]
  )

  const handlePredict = async () => {
    if (!selectedFile) return
    setIsLoading(true)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      })

      const result = await response.json()
      setPrediction(result.prediction)
    } catch (err) {
      console.error("Prediction failed:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const diseaseKey = prediction?.predictedClass
  const disease = diseaseKey ? diseaseDetails[diseaseKey] : null

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-primary mb-4">Derma-AI Skin Lesion Classifier</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Upload an image of a skin lesion to get AI-based classification with expert-backed information and treatments.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-primary">
              <Upload className="h-5 w-5" />
              Upload Image
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div
              className="border-2 border-dashed border-primary/40 rounded-lg p-8 text-center cursor-pointer transition hover:border-primary"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              onClick={() => document.getElementById("file-input")?.click()}
            >
              {previewUrl ? (
                <div className="space-y-4">
                  <img
                    src={previewUrl}
                    alt="Selected lesion"
                    className="max-h-64 mx-auto rounded-lg shadow-md"
                  />
                  <p className="text-sm text-muted-foreground">{selectedFile?.name}</p>
                </div>
              ) : (
                <div className="space-y-4 text-muted-foreground">
                  <ImageIcon className="h-12 w-12 text-primary mx-auto" />
                  <p className="text-lg font-semibold">Drag and drop or click to upload</p>
                  <p className="text-sm">Supported formats: JPG, PNG</p>
                </div>
              )}
            </div>
            <input
              id="file-input"
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />

            <Button
              onClick={handlePredict}
              disabled={!selectedFile || isLoading}
              className="w-full mt-4 bg-primary text-white"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Analyzing...
                </>
              ) : (
                "Predict"
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-primary">Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="py-12 text-center text-muted-foreground">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2 text-blue-500" />
                Processing your image...
              </div>
            ) : disease ? (
              <div className="space-y-6">
                <div className="bg-background p-4 rounded-lg border border-primary/30">
                  <h3 className="text-lg font-semibold text-primary mb-2">Predicted Class</h3>
                  <p className="text-xl font-bold text-blue-500">{disease.name}</p>
                </div>

                <div className="bg-background p-4 rounded-lg border border-green-600/30">
                  <h3 className="text-lg font-semibold text-green-500 mb-2">Confidence</h3>
                  <p className="text-xl font-bold text-green-400">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>

                <div className="bg-background p-4 rounded-lg border border-yellow-400/30">
                  <h3 className="text-lg font-semibold text-yellow-500 mb-2">Disease Information</h3>
                  <p className="text-muted-foreground leading-relaxed">{disease.description}</p>
                </div>

                <div className="bg-background p-4 rounded-lg border border-red-400/30">
                  <h3 className="text-lg font-semibold text-red-500 mb-2">Suggested Treatments</h3>
                  <p className="text-muted-foreground leading-relaxed">{disease.treatment}</p>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-muted-foreground">
                Upload an image and click "Predict" to see results.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
