# NeuroFlow - Cognitive Healthcare Support System

A research-backed, AI-powered Progressive Web App designed to support neurodivergent adults in managing daily routines, emotional well-being, and executive function.

## 🌟 Features

- **Behavioral Pattern Analysis** - AI-powered insights into daily routines and productivity
- **Adaptive Task Management** - Smart task chunking with personalized planning
- **Emotion Recognition** - Multi-modal emotion detection with real-time feedback
- **Interactive Dashboard** - Beautiful, accessible data visualization
- **Smart Notifications** - Context-aware reminders and gentle nudges
- **Cognitive Training** - Gamified exercises with adaptive difficulty
- **Customizable UI** - Sensory-friendly themes and accessibility options

## 🏗️ Project Structure

```
/client          # React PWA Frontend (TypeScript)
  /src
    /components  # Reusable UI components
    /pages       # Main application pages
    /hooks       # Custom React hooks
    /services    # API and external service integrations
    /styles      # Theme system and global styles
    /utils       # Helper functions and utilities
  /public        # Static assets and PWA manifest

/server          # FastAPI Backend (Python)
  /app
    /api         # API endpoints
    /models      # Database models and schemas
    /ml          # Machine learning models and processing
    /core        # Core functionality and configuration

/common          # Shared types and schemas
/docs            # Documentation and research
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+
- MongoDB (local or Atlas)

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/neuroflow.git
cd neuroflow
```

2. Set up the client
```bash
cd client
npm install
npm start
```

3. Set up the server
```bash
cd server
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 🎨 Design Philosophy

NeuroFlow is designed with neurodivergent users in mind:
- **Calming, soothing color palettes**
- **Minimal visual clutter**
- **Smooth, gentle animations**
- **High contrast options**
- **Customizable sensory settings**
- **Distraction-free interface**

## 🔒 Privacy & Ethics

- All emotion recognition features are opt-in
- Data is encrypted and stored securely
- Users can export or delete all their data
- No data sharing without explicit consent
- Open-source and transparent

## 📱 PWA Features

- Installable on desktop and mobile
- Offline functionality for core features
- Push notifications for gentle reminders
- Fast, responsive performance

## 🤝 Contributing

We welcome contributions! Please read our contributing guidelines and code of conduct.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with love for the neurodivergent community, based on research and real user needs.