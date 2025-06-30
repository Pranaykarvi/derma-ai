"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Code, Download, Upload, BarChart3, Shield, Zap } from "lucide-react"

export default function DocumentationPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-foreground mb-2">Documentation</h1>
        <p className="text-lg text-muted-foreground">Complete guide to using Derma-AI for skin lesion classification</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar Navigation */}
        <aside className="lg:col-span-1">
          <Card className="sticky top-4">
            <CardHeader>
              <CardTitle className="text-lg">Quick Navigation</CardTitle>
            </CardHeader>
            <CardContent>
              <nav className="space-y-2 text-sm">
                {[
                  "Getting Started",
                  "API Reference",
                  "Model Details",
                  "Integration Guide",
                  "Code Examples",
                  "Troubleshooting"
                ].map((label) => (
                  <a
                    key={label}
                    href={`#${label.toLowerCase().replace(/\s/g, "-")}`}
                    className="block text-muted-foreground hover:text-primary transition"
                  >
                    {label}
                  </a>
                ))}
              </nav>
            </CardContent>
          </Card>
        </aside>

        {/* Main Content */}
        <main className="lg:col-span-3 space-y-8">
          {/* Getting Started */}
          <section id="getting-started">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-blue-600" />
                  Getting Started
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground">
                  Derma-AI provides a simple web interface for skin lesion classification. Follow these steps to begin:
                </p>
                {[
                  {
                    title: "Upload Image",
                    description: "Upload a dermatoscopic image using the drag-and-drop interface"
                  },
                  {
                    title: "Run Prediction",
                    description: 'Click the "Predict" button to analyze the image'
                  },
                  {
                    title: "View Results",
                    description: "Review classification results and confidence scores"
                  }
                ].map((step, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="bg-primary text-primary-foreground rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold">{i + 1}</span>
                    <div>
                      <h4 className="font-semibold">{step.title}</h4>
                      <p className="text-sm text-muted-foreground">{step.description}</p>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </section>

          {/* API Reference */}
          <section id="api-reference">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Code className="h-5 w-5 text-green-600" />
                  API Reference
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">POST /api/predict</h4>
                  <p className="text-muted-foreground mb-3">Submit an image for classification</p>

                  <div className="bg-muted p-4 rounded-lg">
                    <h5 className="font-semibold mb-2">Request</h5>
                    <pre className="text-sm overflow-x-auto">{`Content-Type: multipart/form-data

{
  "image": <file>
}`}</pre>
                  </div>

                  <div className="bg-muted p-4 rounded-lg mt-4">
                    <h5 className="font-semibold mb-2">Response</h5>
                    <pre className="text-sm overflow-x-auto">{`{
  "prediction": {
    "class": "melanoma",
    "confidence": 0.87,
    "probabilities": {
      "melanoma": 0.87,
      "nevus": 0.08,
      "basal_cell_carcinoma": 0.03,
      "actinic_keratosis": 0.02
    },
    "grad_cam_url": "https://api.derma-ai.com/gradcam/abc123.png"
  },
  "processing_time": 1.2,
  "model_version": "v2.1.0"
}`}</pre>
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          {/* Model Details */}
          <section id="model-details">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-purple-600" />
                  Model Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    {
                      title: "Architecture",
                      items: [
                        "ViT + EfficientNet Fusion",
                        "Input Size: 224×224",
                        "Parameters: 120M",
                        "Pre-trained on ImageNet"
                      ]
                    },
                    {
                      title: "Performance",
                      items: [
                        "Accuracy: 89.9%",
                        "F1-Score: 0.9766",
                        "AUC-ROC: 0.9766",
                        "Training F1: 0.9999"
                      ]
                    }
                  ].map((block, i) => (
                    <div key={i} className="bg-muted p-4 rounded-lg">
                      <h4 className="font-semibold mb-2">{block.title}</h4>
                      <ul className="text-sm text-muted-foreground space-y-1">
                        {block.items.map((item, j) => <li key={j}>• {item}</li>)}
                      </ul>
                    </div>
                  ))}
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Supported Classes</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {[
                      "Melanoma",
                      "Melanocytic Nevus",
                      "Basal Cell Carcinoma",
                      "Actinic Keratosis",
                      "Benign Keratosis",
                      "Dermatofibroma",
                      "Vascular Lesion"
                    ].map((c, i) => (
                      <div key={i} className="bg-primary/10 text-primary px-3 py-1 rounded text-sm text-center">{c}</div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </section>

          {/* Integration Guide */}
          <section id="integration">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5 text-orange-600" />
                  Integration Guide
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {[
                  {
                    label: "JavaScript/React",
                    code: `const formData = new FormData();
formData.append('image', imageFile);
const res = await fetch('/api/predict', {
  method: 'POST',
  body: formData
});
const result = await res.json();`
                  },
                  {
                    label: "Python (requests)",
                    code: `import requests

with open("lesion.jpg", "rb") as img:
  res = requests.post("https://api.derma-ai.com/predict", files={"image": img})
print(res.json())`
                  }
                ].map((item, i) => (
                  <div key={i}>
                    <h4 className="font-semibold mb-1">{item.label} Integration</h4>
                    <div className="bg-muted p-4 rounded-lg overflow-auto">
                      <pre className="text-sm">{item.code}</pre>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </section>

          {/* Code Examples */}
          <section id="examples">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Download className="h-5 w-5 text-indigo-600" />
                  Code Examples
                </CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  ["React Component", "Complete image upload + prediction UI"],
                  ["Python Script", "Batch classify skin images"],
                  ["Node.js Server", "Express API integration"],
                  ["Mobile App", "React Native SDK usage"]
                ].map(([title, desc], i) => (
                  <Button key={i} variant="outline" className="p-4 h-auto flex flex-col items-start bg-transparent">
                    <h4 className="font-semibold mb-1">{title}</h4>
                    <p className="text-sm text-muted-foreground">{desc}</p>
                  </Button>
                ))}
              </CardContent>
            </Card>
          </section>

          {/* Troubleshooting */}
          <section id="troubleshooting">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-red-600" />
                  Troubleshooting
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h4 className="font-semibold mb-2">Common Issues</h4>
                  {[
                    ["Image Upload Fails", "Ensure image is under 10MB and in JPG/PNG format"],
                    ["Low Confidence", "Ensure good image lighting and focus"],
                    ["API Rate Limit", "Free users are limited to 100 requests/day"]
                  ].map(([issue, tip], i) => (
                    <div key={i} className="border-l-4 pl-4 space-y-1 mb-2">
                      <h5 className="font-medium">{issue}</h5>
                      <p className="text-sm text-muted-foreground">{tip}</p>
                    </div>
                  ))}
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Best Practices</h4>
                  <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                    <li>Use dermatoscopic images when possible</li>
                    <li>Ensure lesion is well-lit and centered</li>
                    <li>Trim excessive hair or glare</li>
                    <li>Always consult a dermatologist</li>
                  </ul>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Support</h4>
                  <p className="text-muted-foreground">Need help? Contact:</p>
                  <ul className="text-sm space-y-1">
                    <li>Email: support@derma-ai.com</li>
                    <li>GitHub: github.com/derma-ai</li>
                    <li>Docs: https://docs.derma-ai.com</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </section>
        </main>
      </div>
    </div>
  )
}
