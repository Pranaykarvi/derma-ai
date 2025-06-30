"use client"

import type React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Database, Settings, Brain, Eye, BarChart3, CheckCircle } from "lucide-react"

interface ApproachStep {
  id: number
  title: string
  description: string
  icon: React.ReactNode
  details: string[]
}

export default function ApproachPage() {
  const approachSteps: ApproachStep[] = [
    {
      id: 1,
      title: "Data Preprocessing",
      description: "A robust image preparation pipeline to ensure quality and generalizability.",
      icon: <Database className="h-6 w-6" />,
      details: [
        "Image resizing to 224×224 pixels",
        "Histogram equalization and normalization",
        "Augmentations: rotation, flip, zoom, color jitter",
        "Stratified Train-Validation-Test split",
      ],
    },
    {
      id: 2,
      title: "Model Architecture",
      description: "Fusion-based deep learning model combining spatial and semantic features.",
      icon: <Brain className="h-6 w-6" />,
      details: [
        "Vision Transformer (ViT) pretrained on ImageNet",
        "EfficientNet-B0 for hierarchical feature extraction",
        "Custom fusion layer with linear projection",
        "Final classification head for 7 skin lesion classes",
      ],
    },
    {
      id: 3,
      title: "Training & Optimization",
      description: "Efficient training with regularization and dynamic learning strategies.",
      icon: <Settings className="h-6 w-6" />,
      details: [
        "5-Fold Stratified Cross-Validation",
        "AdamW optimizer with Cosine Annealing scheduler",
        "CrossEntropy loss function with label smoothing",
        "Early stopping and dropout for regularization",
      ],
    },
    {
      id: 4,
      title: "Explainability Tools",
      description: "Building trust with model transparency and interpretability techniques.",
      icon: <Eye className="h-6 w-6" />,
      details: [
        "Grad-CAM heatmaps to visualize lesion importance",
        "SHAP (SHapley Additive exPlanations) for pixel-wise relevance",
        "Transformer attention map visualization",
        "Class activation overlays for clinical feedback",
      ],
    },
    {
      id: 5,
      title: "Evaluation Methods",
      description: "Thorough validation to ensure clinical relevance and generalizability.",
      icon: <BarChart3 className="h-6 w-6" />,
      details: [
        "Multi-class F1-score and ROC-AUC",
        "Accuracy and confusion matrix per class",
        "Clinical evaluation against real HAM10000 labels",
        "Cross-fold average metrics for robustness",
      ],
    },
    {
      id: 6,
      title: "Deployment & Monitoring",
      description: "Scalable cloud deployment using modern infrastructure for real-world accessibility.",
      icon: <CheckCircle className="h-6 w-6" />,
      details: [
        "Frontend deployed on Vercel for fast CDN-based delivery",
        "Backend REST API hosted on Render.com with auto-scaling",
        "Model served via Flask or FastAPI for prediction endpoints",
        "Live health checks and log-based performance monitoring",
      ],
    },
  ]
  
  

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-foreground mb-4">Our Technical Approach</h1>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
          We designed a deep learning pipeline that fuses transformer and convolutional models to achieve
          state-of-the-art performance in skin lesion classification, while also prioritizing explainability and robustness.
        </p>
      </div>

      <div className="space-y-8">
        {approachSteps.map((step, index) => (
          <div key={step.id} className="relative">
            {index < approachSteps.length - 1 && (
              <div className="absolute left-6 top-20 w-0.5 h-16 bg-primary/20 hidden md:block" />
            )}

            <Card className="relative">
              <CardHeader>
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center text-primary">
                    {step.icon}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="bg-primary text-primary-foreground text-sm font-bold px-2 py-1 rounded">
                        Step {step.id}
                      </span>
                      <CardTitle className="text-xl">{step.title}</CardTitle>
                    </div>
                    <p className="text-muted-foreground">{step.description}</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="ml-16">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {step.details.map((detail, detailIndex) => (
                      <div key={detailIndex} className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                        <span className="text-sm text-foreground">{detail}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ))}
      </div>

      {/* Pipeline Overview */}
      <Card className="mt-12">
        <CardHeader>
          <CardTitle className="text-center text-foreground">End-to-End Pipeline Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gradient-to-r from-primary/5 to-purple-500/5 rounded-lg p-8">
            <div className="flex flex-wrap justify-center items-center gap-4 text-center">
              {approachSteps.map((step, index) => (
                <div key={step.id} className="flex items-center">
                  <div className="bg-background rounded-lg p-3 shadow-sm">
                    <div className="text-primary mb-1">{step.icon}</div>
                    <p className="text-xs font-medium text-foreground">{step.title}</p>
                  </div>
                  {index < approachSteps.length - 1 && (
                    <div className="hidden sm:block w-8 h-0.5 bg-primary/30 mx-2" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
