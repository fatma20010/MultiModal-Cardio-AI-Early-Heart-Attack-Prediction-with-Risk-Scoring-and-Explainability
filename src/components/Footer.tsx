import { Shield } from 'lucide-react';

export const Footer = () => {
  return (
    <footer className="bg-card/80 backdrop-blur-sm border-t border-border mt-20">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Shield className="w-4 h-4" />
            <p>All data is processed securely and confidentially</p>
          </div>
          
          <div className="text-center md:text-right">
            <p className="text-sm text-muted-foreground">
              Powered by <span className="font-semibold text-foreground">Advanced AI Technology</span>
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              © 2025 CardioAI Diagnosis • For medical professional use
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};
