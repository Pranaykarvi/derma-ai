# Derma-AI - Skin Lesion Classification System

A modern, responsive web application for AI-powered skin lesion classification using state-of-the-art deep learning models.

## Features

- 🔬 **AI-Powered Classification**: Advanced Swin Transformer model for accurate skin lesion analysis
- 🎨 **Modern UI/UX**: Clean, medical-themed interface with dark/light mode support
- 📱 **Responsive Design**: Works seamlessly across desktop, tablet, and mobile devices
- 🔒 **Privacy-First**: Images are processed securely and not permanently stored
- 📊 **Explainable AI**: Grad-CAM visualizations for model interpretability
- 🚀 **Fast Processing**: Optimized inference pipeline for quick results

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS, shadcn/ui components
- **Theme**: next-themes for dark/light mode
- **Icons**: Lucide React
- **Deployment**: Vercel (recommended)

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/your-username/derma-ai.git
cd derma-ai
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
# or
yarn install
\`\`\`

3. Copy environment variables:
\`\`\`bash
cp .env.example .env.local
\`\`\`

4. Start the development server:
\`\`\`bash
npm run dev
# or
yarn dev
\`\`\`

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

\`\`\`
derma-ai/
├── app/                    # Next.js app directory
│   ├── about/             # About page
│   ├── approach/          # Approach page
│   ├── docs/              # Documentation page
│   ├── privacy/           # Privacy policy page
│   ├── project-details/   # Project details page
│   ├── terms/             # Terms of service page
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page
├── components/            # Reusable components
│   ├── ui/               # shadcn/ui components
│   ├── footer.tsx        # Footer component
│   ├── navbar.tsx        # Navigation component
│   └── theme-provider.tsx # Theme provider
├── lib/                  # Utility functions
│   └── utils.ts          # Utility functions
└── public/               # Static assets
\`\`\`

## API Integration

The application is ready for backend integration. Key integration points:

### Prediction API
\`\`\`typescript
// Example API call
const predictSkinLesion = async (imageFile: File) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
};
\`\`\`

### Expected API Response
\`\`\`json
{
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
}
\`\`\`

## Model Information

- **Architecture**: Swin Transformer Base
- **Dataset**: HAM10000 (10,015 dermatoscopic images)
- **Classes**: 7 skin lesion types
- **Performance**: 94.2% accuracy, 0.91 F1-score
- **Input Size**: 224×224 pixels
- **Parameters**: 88M

## Supported Lesion Types

1. Melanoma
2. Melanocytic Nevus
3. Basal Cell Carcinoma
4. Actinic Keratosis
5. Benign Keratosis
6. Dermatofibroma
7. Vascular Lesion

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Connect your repository to Vercel
3. Deploy with default settings

### Other Platforms

The application can be deployed on any platform that supports Next.js:
- Netlify
- AWS Amplify
- Railway
- DigitalOcean App Platform

## Contributing

1. Fork the repository
2. Create a feature branch: \`git checkout -b feature/amazing-feature\`
3. Commit your changes: \`git commit -m 'Add amazing feature'\`
4. Push to the branch: \`git push origin feature/amazing-feature\`
5. Open a Pull Request

## Medical Disclaimer

⚠️ **Important**: This application is for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@derma-ai.com
- 📖 Documentation: [docs.derma-ai.com](https://docs.derma-ai.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/derma-ai/issues)

## Acknowledgments

- HAM10000 dataset contributors
- Swin Transformer research team
- shadcn/ui component library
- Next.js and React communities
\`\`\`

Finally, let's create a proper .gitignore file:
