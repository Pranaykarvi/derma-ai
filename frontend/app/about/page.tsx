"use client"

import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Github, Linkedin, Mail, User, Globe } from "lucide-react"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faKaggle } from "@fortawesome/free-brands-svg-icons"

interface TeamMember {
  id: number
  name: string
  role: string
  bio: string
  imageUrl: string
  linkedinUrl?: string
  githubUrl?: string
  email?: string
  kaggleUrl?: string
}

export default function AboutPage() {
  const teamMembers: TeamMember[] = [
    {
      id: 1,
      name: "Pranay Kumar Karvi",
      role: "PreFinal Year at VIT Chennai",
      bio: "Focused on building explainable AI systems for healthcare using cutting-edge deep learning and radiomics. Final-year Data Science student at VIT.",
      imageUrl: "/placeholder.svg?height=200&width=200",
      linkedinUrl: "https://linkedin.com/in/pranaykarvi",
      githubUrl: "https://github.com/pranaykarvi",
      email: "pranaykumar.karvi2023@vitstudent.ac.in",
      kaggleUrl: "https://www.kaggle.com/pranaykarvi",
    },
    {
      id: 2,
      name: "Samriddhi Ganguly",
      role: "PreFinal Year at VIT Chennai",
      bio: "Passionate about medical AI, full-stack development, and deploying scalable ML solutions. Also pursuing Data Science at Vellore Institute of Technology.",
      imageUrl: "/placeholder.svg?height=200&width=200",
      linkedinUrl: "https://www.linkedin.com/in/samriddhi-ganguly-2b173929a/",
      githubUrl: "https://github.com/sammmmmyyyy",
      email: "samriddhi.ganguly05@gmail.com",
      kaggleUrl: "https://kaggle.com/sammganguly05",
    },
  ]

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-foreground mb-4">About Our Team</h1>
        <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
          We are a dedicated team of final-year students at Vellore Institute of Technology, specializing in Data
          Science. Our mission is to revolutionize skin lesion diagnosis with explainable, reliable, and accessible
          AI.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-8 mb-12">
        {teamMembers.map((member) => (
          <Card key={member.id} className="overflow-hidden hover:shadow-lg transition-shadow">
            <CardContent className="p-6">
              <div className="flex flex-col items-center text-center">
                <div className="relative mb-4">
                  <img
                    src={member.imageUrl || "/placeholder.svg"}
                    alt={member.name}
                    className="w-32 h-32 rounded-full object-cover border-4 border-primary/20"
                  />
                  <div className="absolute inset-0 w-32 h-32 rounded-full bg-primary/10 flex items-center justify-center">
                    <User className="h-16 w-16 text-primary/60" />
                  </div>
                </div>

                <h3 className="text-xl font-bold text-foreground mb-1">{member.name}</h3>
                <p className="text-primary font-medium mb-3">{member.role}</p>
                <p className="text-muted-foreground text-sm mb-4 leading-relaxed">{member.bio}</p>

                <div className="flex gap-2">
                  {member.linkedinUrl && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="p-2 bg-transparent"
                      onClick={() => window.open(member.linkedinUrl, "_blank")}
                    >
                      <Linkedin className="h-4 w-4" />
                    </Button>
                  )}
                  {member.githubUrl && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="p-2 bg-transparent"
                      onClick={() => window.open(member.githubUrl, "_blank")}
                    >
                      <Github className="h-4 w-4" />
                    </Button>
                  )}
                  {member.email && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="p-2 bg-transparent"
                      onClick={() => window.open(`mailto:${member.email}`, "_blank")}
                    >
                      <Mail className="h-4 w-4" />
                    </Button>
                  )}
                  {member.kaggleUrl && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="p-2 bg-transparent"
                      onClick={() => window.open(member.kaggleUrl, "_blank")}
                    >
                      <FontAwesomeIcon icon={faKaggle} className="h-4 w-4 text-[#20BEFF]" />
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="bg-gradient-to-r from-primary/5 to-purple-500/5">
        <CardContent className="p-8 text-center">
          <h2 className="text-2xl font-bold text-foreground mb-4">Our Mission</h2>
          <p className="text-foreground max-w-3xl mx-auto leading-relaxed">
            To democratize access to dermatological screening through powerful AI solutions that prioritize accuracy,
            transparency, and accessibility—enabling early detection and better healthcare outcomes for everyone.
          </p>
        </CardContent>
      </Card>

      <div className="mt-12 text-center">
        <Card>
          <CardContent className="p-8">
            <h2 className="text-2xl font-bold text-foreground mb-4">Get In Touch</h2>
            <p className="text-muted-foreground mb-6">Interested in collaborating or learning more about our work?</p>
            <Button className="bg-primary hover:bg-primary/90">
              <Mail className="h-4 w-4 mr-2" />
              Contact Us
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

