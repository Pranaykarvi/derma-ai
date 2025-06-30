"use client"

import Link from "next/link"
import { Github, Linkedin, Mail, Trophy } from "lucide-react"
import { Button } from "@/components/ui/button"

interface TeamMember {
  name: string
  role: string
  github?: string
  linkedin?: string
  kaggle?: string
  email?: string
}

export default function Footer() {
  const teamMembers: TeamMember[] = [
    {
      name: "Pranay Karvi",
      role: "AI Researcher & Developer",
      github: "https://github.com/pranaykarvi",
      linkedin: "https://linkedin.com/in/pranaykarvi",
      kaggle: "https://kaggle.com/pranaykarvi",
      email: "pranaykarvi@gmail.com",
    },
    {
      name: "Contributor 2",
      role: "ML Engineer",
      github: "https://github.com/contributor2",
      linkedin: "https://linkedin.com/in/contributor2",
      kaggle: "https://kaggle.com/contributor2",
      email: "contributor2@vitstudent.ac.in",
    },
  ]

  const quickLinks = [
    { name: "Home", href: "/" },
    { name: "Project Details", href: "/project-details" },
    { name: "About Us", href: "/about" },
    { name: "Documentation", href: "/docs" },
  ]

  return (
    <footer className="bg-background border-t border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Brand Section */}
          <div className="space-y-4">
            <h3 className="text-2xl font-bold text-primary">Derma-AI</h3>
            <p className="text-muted-foreground text-sm leading-relaxed">
              AI-powered skin lesion classification system for early detection and improved healthcare outcomes.
            </p>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm" asChild>
                <Link href="mailto:contact@derma-ai.com">
                  <Mail className="h-4 w-4 mr-2" />
                  Contact
                </Link>
              </Button>
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-foreground">Quick Links</h4>
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.name}>
                  <Link
                    href={link.href}
                    className="text-muted-foreground hover:text-primary transition-colors text-sm"
                  >
                    {link.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Team Members */}
          <div className="space-y-4">
            <h4 className="text-lg font-semibold text-foreground">Our Team</h4>
            <div className="space-y-4">
              {teamMembers.map((member, index) => (
                <div key={index} className="space-y-1">
                  <h5 className="font-medium text-foreground text-sm">{member.name}</h5>
                  <p className="text-xs text-muted-foreground">{member.role}</p>
                  <div className="flex space-x-1">
                    {member.github && (
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" asChild>
                        <Link href={member.github} target="_blank" rel="noopener noreferrer">
                          <Github className="h-3 w-3" />
                          <span className="sr-only">GitHub</span>
                        </Link>
                      </Button>
                    )}
                    {member.linkedin && (
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" asChild>
                        <Link href={member.linkedin} target="_blank" rel="noopener noreferrer">
                          <Linkedin className="h-3 w-3" />
                          <span className="sr-only">LinkedIn</span>
                        </Link>
                      </Button>
                    )}
                    {member.kaggle && (
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" asChild>
                        <Link href={member.kaggle} target="_blank" rel="noopener noreferrer">
                          <Trophy className="h-3 w-3" />
                          <span className="sr-only">Kaggle</span>
                        </Link>
                      </Button>
                    )}
                    {member.email && (
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0" asChild>
                        <Link href={`mailto:${member.email}`}>
                          <Mail className="h-3 w-3" />
                          <span className="sr-only">Email</span>
                        </Link>
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="mt-8 pt-8 border-t border-border">
          <div className="flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} Derma-AI. All rights reserved.
            </p>
            <div className="flex space-x-6">
              <Link
                href="/privacy"
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                Privacy Policy
              </Link>
              <Link
                href="/terms"
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                Terms of Service
              </Link>
              <Link
                href="/docs"
                className="text-sm text-muted-foreground hover:text-primary transition-colors"
              >
                Documentation
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}
