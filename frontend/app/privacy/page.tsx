import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function PrivacyPolicyPage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-foreground mb-4">Privacy Policy</h1>
        <p className="text-muted-foreground">Last updated: {new Date().toLocaleDateString()}</p>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>1. Information We Collect</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Medical Images</h4>
              <p className="text-muted-foreground">
                When you upload skin lesion images for analysis, we temporarily process these images through our AI
                model. Images are not permanently stored on our servers and are deleted after processing is complete.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Usage Data</h4>
              <p className="text-muted-foreground">
                We collect anonymous usage statistics to improve our service, including prediction accuracy metrics and
                system performance data. This data cannot be used to identify individual users.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Technical Information</h4>
              <p className="text-muted-foreground">
                We automatically collect certain technical information such as IP addresses, browser type, and device
                information for security and optimization purposes.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2. How We Use Your Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>To provide AI-powered skin lesion classification services</li>
              <li>To improve the accuracy and performance of our AI models</li>
              <li>To ensure the security and proper functioning of our platform</li>
              <li>To comply with legal obligations and regulatory requirements</li>
              <li>To communicate with users about service updates and important notices</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>3. Data Security</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              We implement industry-standard security measures to protect your data, including encryption in transit and
              at rest, secure data processing protocols, and regular security audits. All medical images are processed
              in secure, HIPAA-compliant environments and are automatically deleted after analysis.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>4. Data Sharing</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              We do not sell, trade, or otherwise transfer your personal information to third parties. We may share
              anonymized, aggregated data for research purposes or to improve medical AI technologies, but this data
              cannot be used to identify individual users.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>5. Your Rights</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Right to access information about how your data is processed</li>
              <li>Right to request deletion of your data (where applicable)</li>
              <li>Right to opt-out of data collection for research purposes</li>
              <li>Right to receive information about data breaches that may affect you</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>6. Medical Disclaimer</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Derma-AI is a research tool and should not be used as a substitute for professional medical advice,
              diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns. Our
              AI predictions are for informational purposes only and should not be relied upon for medical decisions.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>7. Contact Information</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              If you have questions about this Privacy Policy or our data practices, please contact us at:
            </p>
            <div className="mt-4 space-y-1">
              <p className="text-foreground">Email: privacy@derma-ai.com</p>
              <p className="text-foreground">Address: [Your Address]</p>
              <p className="text-foreground">Phone: [Your Phone Number]</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
