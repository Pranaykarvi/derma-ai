import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function TermsOfServicePage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-foreground mb-4">Terms of Service</h1>
        <p className="text-muted-foreground">Last updated: {new Date().toLocaleDateString()}</p>
      </div>

      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>1. Acceptance of Terms</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              By accessing and using Derma-AI, you accept and agree to be bound by the terms and provision of this
              agreement. If you do not agree to abide by the above, please do not use this service.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2. Service Description</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              Derma-AI provides AI-powered skin lesion classification services for research and educational purposes.
              Our platform uses machine learning algorithms to analyze dermatoscopic images and provide classification
              predictions.
            </p>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">Important Medical Disclaimer</h4>
              <p className="text-yellow-700 dark:text-yellow-300 text-sm">
                This service is NOT intended for medical diagnosis or treatment decisions. Always consult qualified
                healthcare professionals for medical advice.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>3. User Responsibilities</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <h4 className="font-semibold">You agree to:</h4>
            <ul className="list-disc list-inside space-y-2 text-muted-foreground">
              <li>Use the service only for lawful purposes and in accordance with these Terms</li>
              <li>Not upload images that violate privacy rights or contain inappropriate content</li>
              <li>Not attempt to reverse engineer, hack, or compromise the security of our systems</li>
              <li>Not use the service for commercial purposes without explicit permission</li>
              <li>Understand that AI predictions are not medical diagnoses</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>4. Intellectual Property</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              The Derma-AI platform, including its AI models, algorithms, software, and content, is protected by
              intellectual property laws. You may not copy, modify, distribute, or create derivative works without
              explicit written permission.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>5. Limitation of Liability</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              Derma-AI and its developers shall not be liable for any direct, indirect, incidental, special,
              consequential, or punitive damages resulting from your use of the service.
            </p>
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
              <h4 className="font-semibold text-red-800 dark:text-red-200 mb-2">Medical Liability Disclaimer</h4>
              <p className="text-red-700 dark:text-red-300 text-sm">
                We are not liable for any medical decisions made based on our AI predictions. This tool is for research
                and educational purposes only.
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>6. Service Availability</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              We strive to maintain high service availability but do not guarantee uninterrupted access. The service may
              be temporarily unavailable due to maintenance, updates, or technical issues.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>7. Privacy and Data Protection</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Your privacy is important to us. Please review our Privacy Policy to understand how we collect, use, and
              protect your information. By using our service, you consent to our data practices as described in the
              Privacy Policy.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>8. Modifications to Terms</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              We reserve the right to modify these terms at any time. Users will be notified of significant changes, and
              continued use of the service constitutes acceptance of the modified terms.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>9. Termination</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              We may terminate or suspend access to our service immediately, without prior notice, for any reason
              whatsoever, including without limitation if you breach the Terms.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>10. Contact Information</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              If you have any questions about these Terms of Service, please contact us at:
            </p>
            <div className="mt-4 space-y-1">
              <p className="text-foreground">Email: legal@derma-ai.com</p>
              <p className="text-foreground">Address: [Your Address]</p>
              <p className="text-foreground">Phone: [Your Phone Number]</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
