import React from "react";
import { useThemeStore } from "../hooks/useWorkingTheme";
import {
  Home,
  CheckSquare,
  BarChart3,
  Settings,
  Moon,
  Sun,
  Brain,
  Heart,
  Zap,
  Smile,
} from "lucide-react";
import { useLocation, Link } from "react-router-dom";

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { darkMode, setDarkMode } = useThemeStore();
  const location = useLocation();

  const navItems = [
    { icon: Home, label: "Dashboard", path: "/dashboard" },
    { icon: CheckSquare, label: "Tasks", path: "/tasks" },
    { icon: Smile, label: "Emotion Analysis", path: "/emotion-analysis" },
    { icon: BarChart3, label: "Insights", path: "/insights" },
    { icon: Settings, label: "Customize UI", path: "/settings" },
  ];

  const isActive = (path: string) =>
    location.pathname === path ||
    (path === "/dashboard" && location.pathname === "/");

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  return (
    <div className="min-h-screen bg-background transition-colors duration-200">
      {/* Header */}
      <header className="bg-surface shadow-sm border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="bg-primary rounded-lg p-2">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <h1 className="text-xl font-semibold text-text-primary">
                NeuroFlow
              </h1>
            </div>

            {/* Navigation - Desktop */}
            <nav className="hidden md:flex space-x-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                      isActive(item.path)
                        ? "bg-primary/10 text-primary border border-primary/20"
                        : "text-text-secondary hover:bg-surface hover:text-text-primary border border-transparent"
                    }`}
                  >
                    <Icon size={18} />
                    <span className="font-medium">{item.label}</span>
                  </Link>
                );
              })}
            </nav>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg bg-surface border border-border hover:bg-primary/10 transition-colors duration-200"
              aria-label="Toggle theme"
            >
              {darkMode ? (
                <Sun size={20} className="text-text-primary" />
              ) : (
                <Moon size={20} className="text-text-primary" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Mobile Navigation */}
      <nav className="md:hidden bg-surface border-t border-border fixed bottom-0 left-0 right-0 z-50">
        <div className="flex justify-around items-center py-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex flex-col items-center space-y-1 p-2 rounded-lg transition-all duration-200 ${
                  isActive(item.path) ? "text-primary" : "text-text-secondary"
                }`}
              >
                <Icon size={20} />
                <span className="text-xs font-medium">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
};
