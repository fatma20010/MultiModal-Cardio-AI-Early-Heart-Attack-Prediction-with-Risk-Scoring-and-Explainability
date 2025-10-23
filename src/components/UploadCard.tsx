import { useRef, useState } from 'react';
import { Upload, X, FileImage, Music, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface UploadCardProps {
  type: 'ecg' | 'pcg';
  title: string;
  description: string;
  accept: string;
  maxSize: number;
  file: File | null;
  onFileChange: (file: File | null) => void;
}

export const UploadCard = ({
  type,
  title,
  description,
  accept,
  maxSize,
  file,
  onFileChange,
}: UploadCardProps) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string>('');

  const borderColor = type === 'ecg' ? 'border-primary' : 'border-secondary';
  const bgColor = type === 'ecg' ? 'bg-primary/5' : 'bg-secondary/5';
  const Icon = type === 'ecg' ? FileImage : Music;

  const validateFile = (file: File): boolean => {
    if (file.size > maxSize * 1024 * 1024) {
      setError(`File size must be less than ${maxSize}MB`);
      return false;
    }
    
    const acceptedTypes = accept.split(',').map(t => t.trim());
    const fileType = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!acceptedTypes.includes(fileType)) {
      setError(`Please upload a ${accept} file`);
      return false;
    }
    
    setError('');
    return true;
  };

  const handleFile = (selectedFile: File) => {
    if (validateFile(selectedFile)) {
      onFileChange(selectedFile);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  return (
    <div className={cn(
      "gradient-card rounded-2xl p-6 border-2 transition-smooth shadow-medical",
      dragActive ? `${borderColor} shadow-glow` : 'border-border',
      file && 'border-success'
    )}>
      <div className="flex items-center gap-3 mb-4">
        <div className={cn("p-3 rounded-xl", bgColor)}>
          <Icon className={cn(
            "w-6 h-6",
            type === 'ecg' ? 'text-primary' : 'text-secondary'
          )} />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </div>

      <div
        className={cn(
          "border-2 border-dashed rounded-xl p-8 text-center transition-smooth cursor-pointer",
          dragActive ? `${borderColor} ${bgColor}` : 'border-border hover:border-primary/50',
          file && 'bg-success/5 border-success'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          accept={accept}
          onChange={handleChange}
        />
        
        {file ? (
          <div className="space-y-3">
            <CheckCircle className="w-12 h-12 text-success mx-auto" />
            <div>
              <p className="font-medium text-foreground">{file.name}</p>
              <p className="text-sm text-muted-foreground">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onFileChange(null);
              }}
              className="mt-2"
            >
              <X className="w-4 h-4 mr-2" />
              Remove
            </Button>
          </div>
        ) : (
          <div className="space-y-3">
            <Upload className="w-12 h-12 text-muted-foreground mx-auto" />
            <div>
              <p className="font-medium text-foreground">Drop your file here or click to browse</p>
              <p className="text-sm text-muted-foreground mt-1">
                {accept} • Max {maxSize}MB
              </p>
            </div>
          </div>
        )}
      </div>
      
      {error && (
        <p className="text-sm text-danger mt-2">{error}</p>
      )}
    </div>
  );
};
