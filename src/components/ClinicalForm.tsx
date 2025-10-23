import { useForm } from 'react-hook-form';
import { Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tooltip } from 'react-tooltip';
import { ClinicalData } from '@/services/api';

interface ClinicalFormProps {
  onSubmit: (data: ClinicalData) => void;
  isLoading: boolean;
}

export const ClinicalForm = ({ onSubmit, isLoading }: ClinicalFormProps) => {
  const { register, handleSubmit, setValue, formState: { errors } } = useForm<ClinicalData>({
    defaultValues: {
      temp: 98.6,
    }
  });

  const FormField = ({ 
    name, 
    label, 
    tooltip, 
    type = 'number',
    min,
    max,
    step,
    required = true
  }: {
    name: keyof ClinicalData;
    label: string;
    tooltip: string;
    type?: string;
    min?: number;
    max?: number;
    step?: number;
    required?: boolean;
  }) => (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label htmlFor={name} className="text-foreground font-medium">
          {label} {required && <span className="text-danger">*</span>}
        </Label>
        <Info 
          className="w-4 h-4 text-muted-foreground cursor-help" 
          data-tooltip-id={name}
          data-tooltip-content={tooltip}
        />
        <Tooltip id={name} />
      </div>
      <Input
        id={name}
        type={type}
        min={min}
        max={max}
        step={step}
        {...register(name, { 
          required: required ? `${label} is required` : false,
          valueAsNumber: type === 'number',
          min: min ? { value: min, message: `Minimum value is ${min}` } : undefined,
          max: max ? { value: max, message: `Maximum value is ${max}` } : undefined,
        })}
        className="bg-card border-border focus:border-primary"
      />
      {errors[name] && (
        <p className="text-sm text-danger">{errors[name]?.message}</p>
      )}
    </div>
  );

  const SelectField = ({
    name,
    label,
    tooltip,
    options,
    required = true
  }: {
    name: keyof ClinicalData;
    label: string;
    tooltip: string;
    options: Array<{ value: number; label: string }>;
    required?: boolean;
  }) => (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Label htmlFor={name} className="text-foreground font-medium">
          {label} {required && <span className="text-danger">*</span>}
        </Label>
        <Info 
          className="w-4 h-4 text-muted-foreground cursor-help" 
          data-tooltip-id={name}
          data-tooltip-content={tooltip}
        />
        <Tooltip id={name} />
      </div>
      <Select onValueChange={(value) => setValue(name, parseInt(value))}>
        <SelectTrigger className="bg-card border-border">
          <SelectValue placeholder={`Select ${label.toLowerCase()}`} />
        </SelectTrigger>
        <SelectContent>
          {options.map(opt => (
            <SelectItem key={opt.value} value={opt.value.toString()}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {errors[name] && (
        <p className="text-sm text-danger">{errors[name]?.message}</p>
      )}
    </div>
  );

  return (
    <div className="gradient-card rounded-2xl p-8 shadow-medical border border-border">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-accent/20 p-3 rounded-xl">
          <Info className="w-6 h-6 text-accent" />
        </div>
        <div>
          <h3 className="text-xl font-semibold text-foreground">Clinical Data</h3>
          <p className="text-sm text-muted-foreground">Please provide accurate patient information</p>
        </div>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <FormField
            name="age"
            label="Age"
            tooltip="Patient's age in years"
            min={0}
            max={120}
          />
          
          <SelectField
            name="sex"
            label="Sex"
            tooltip="Biological sex"
            options={[
              { value: 0, label: 'Female' },
              { value: 1, label: 'Male' }
            ]}
          />
          
          <SelectField
            name="cp"
            label="Chest Pain Type"
            tooltip="Type of chest pain experienced"
            options={[
              { value: 0, label: 'Typical Angina' },
              { value: 1, label: 'Atypical Angina' },
              { value: 2, label: 'Non-Anginal Pain' },
              { value: 3, label: 'Asymptomatic' }
            ]}
          />
          
          <FormField
            name="trtbps"
            label="Resting BP (mm Hg)"
            tooltip="Resting blood pressure in mm Hg"
            min={80}
            max={200}
          />
          
          <FormField
            name="chol"
            label="Cholesterol (mg/dl)"
            tooltip="Serum cholesterol in mg/dl"
            min={100}
            max={600}
          />
          
          <SelectField
            name="fbs"
            label="Fasting Blood Sugar"
            tooltip="Fasting blood sugar > 120 mg/dl"
            options={[
              { value: 0, label: 'False (<120)' },
              { value: 1, label: 'True (>120)' }
            ]}
          />
          
          <SelectField
            name="restecg"
            label="Resting ECG"
            tooltip="Resting electrocardiographic results"
            options={[
              { value: 0, label: 'Normal' },
              { value: 1, label: 'ST-T Wave Abnormality' },
              { value: 2, label: 'Left Ventricular Hypertrophy' }
            ]}
          />
          
          <FormField
            name="thalachh"
            label="Max Heart Rate"
            tooltip="Maximum heart rate achieved"
            min={60}
            max={220}
          />
          
          <SelectField
            name="exng"
            label="Exercise Angina"
            tooltip="Exercise induced angina"
            options={[
              { value: 0, label: 'No' },
              { value: 1, label: 'Yes' }
            ]}
          />
          
          <FormField
            name="oldpeak"
            label="ST Depression"
            tooltip="ST depression induced by exercise"
            min={0}
            max={5}
            step={0.1}
          />
          
          <SelectField
            name="slp"
            label="ST Slope"
            tooltip="Slope of peak exercise ST segment"
            options={[
              { value: 0, label: 'Upsloping' },
              { value: 1, label: 'Flat' },
              { value: 2, label: 'Downsloping' }
            ]}
          />
          
          <SelectField
            name="caa"
            label="Major Vessels"
            tooltip="Number of major vessels colored by fluoroscopy"
            options={[
              { value: 0, label: '0' },
              { value: 1, label: '1' },
              { value: 2, label: '2' },
              { value: 3, label: '3' }
            ]}
          />
          
          <SelectField
            name="thall"
            label="Thalassemia"
            tooltip="Thalassemia test result"
            options={[
              { value: 0, label: 'Normal' },
              { value: 1, label: 'Fixed Defect' },
              { value: 2, label: 'Reversible Defect' },
              { value: 3, label: 'Not Described' }
            ]}
          />
          
          <FormField
            name="temp"
            label="Temperature (°F)"
            tooltip="Body temperature in Fahrenheit"
            min={95}
            max={105}
            step={0.1}
          />
        </div>

        <div className="flex justify-end pt-6 border-t border-border">
          <Button
            type="submit"
            size="lg"
            disabled={isLoading}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-8 rounded-xl shadow-medical transition-smooth hover:shadow-glow"
          >
            {isLoading ? 'Analyzing...' : 'Analyze Risk'}
          </Button>
        </div>
      </form>
    </div>
  );
};
