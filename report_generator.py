from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os

class CardiacReportGenerator:
    """Generate PDF reports for cardiac abnormality predictions"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2563eb'),
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskLevel',
            parent=self.styles['Normal'],
            fontSize=32,
            spaceBefore=10,
            spaceAfter=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
    
    def _get_risk_color(self, level):
        """Get color based on risk level"""
        if level <= 2:
            return colors.HexColor('#10b981')  # Green
        elif level == 3:
            return colors.HexColor('#f59e0b')  # Amber
        else:
            return colors.HexColor('#ef4444')  # Red
    
    def generate_report(self, prediction_data, output_path='cardiac_report.pdf'):
        """
        Generate comprehensive PDF report
        
        Args:
            prediction_data: Dictionary with prediction results
            output_path: Path to save the PDF
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        story = []
        
        # Header
        story.append(Paragraph("🫀 Cardiac Abnormality Assessment Report", 
                              self.styles['CustomTitle']))
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        date_str = datetime.now().strftime("%B %d, %Y at %H:%M")
        story.append(Paragraph(f"<b>Report Generated:</b> {date_str}", 
                              self.styles['Normal']))
        story.append(Paragraph("<b>Analysis Type:</b> Multimodal AI-Powered Assessment", 
                              self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Risk Assessment Summary Box
        story.append(Paragraph("Risk Assessment Summary", self.styles['SectionHeader']))
        
        risk_level = prediction_data['risk_assessment']['level']
        risk_category = prediction_data['risk_assessment']['category']
        probability = prediction_data['probability']
        
        # Create risk summary table
        risk_color = self._get_risk_color(risk_level)
        risk_data = [
            ['Cardiac Risk Probability', f"{probability*100:.1f}%"],
            ['Risk Level', f"Level {risk_level} of 5"],
            ['Risk Category', risk_category],
        ]
        
        risk_table = Table(risk_data, colWidths=[3*inch, 3*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, 0), risk_color),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 0), (1, 0), 18),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Interpretation
        story.append(Paragraph("Clinical Interpretation", self.styles['SectionHeader']))
        story.append(Paragraph(prediction_data['risk_assessment']['interpretation'], 
                              self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        story.append(Paragraph("Medical Recommendations", self.styles['SectionHeader']))
        story.append(Paragraph(f"<b>Recommendation:</b> {prediction_data['risk_assessment']['recommendation']}", 
                              self.styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"<b>Action Required:</b> {prediction_data['risk_assessment']['action']}", 
                              self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Explainability Section (if available)
        if 'explainability' in prediction_data and prediction_data['explainability']:
            explainability = prediction_data['explainability']
            
            story.append(Paragraph("AI Explainability Analysis", self.styles['SectionHeader']))
            story.append(Spacer(1, 0.1*inch))
            
            # Primary Driver
            story.append(Paragraph(f"<b>Primary Driver:</b> {explainability['primary_driver']}", 
                                  self.styles['Normal']))
            story.append(Paragraph(f"<i>{explainability['explanation']}</i>", 
                                  self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Modality Contributions Table
            story.append(Paragraph("Modality Contributions", self.styles['Heading3']))
            modality_data = [
                ['Modality', 'Contribution', 'Impact', 'Interpretation'],
                ['ECG (Electrocardiogram)', 
                 f"{explainability['modality_contributions']['ecg']['percentage']:.1f}%",
                 f"{explainability['modality_contributions']['ecg']['impact']:.3f}",
                 explainability['modality_contributions']['ecg']['interpretation']],
                ['PCG (Phonocardiogram)', 
                 f"{explainability['modality_contributions']['pcg']['percentage']:.1f}%",
                 f"{explainability['modality_contributions']['pcg']['impact']:.3f}",
                 explainability['modality_contributions']['pcg']['interpretation']],
                ['Clinical Data', 
                 f"{explainability['modality_contributions']['clinical']['percentage']:.1f}%",
                 f"{explainability['modality_contributions']['clinical']['impact']:.3f}",
                 explainability['modality_contributions']['clinical']['interpretation']],
            ]
            
            modality_table = Table(modality_data, colWidths=[1.5*inch, 1*inch, 0.8*inch, 3.2*inch])
            modality_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(modality_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Top Clinical Features
            story.append(Paragraph("Top Clinical Features", self.styles['Heading3']))
            feature_data = [['Feature', 'Importance', 'Interpretation']]
            for feature in explainability['top_clinical_features'][:5]:
                feature_data.append([
                    feature['feature'].upper(),
                    f"{feature['importance']*100:.1f}%",
                    feature['interpretation']
                ])
            
            feature_table = Table(feature_data, colWidths=[1.5*inch, 1.2*inch, 3.8*inch])
            feature_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(feature_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Confidence Assessment
            story.append(Paragraph("Confidence Assessment", self.styles['Heading3']))
            conf = explainability['confidence_assessment']
            story.append(Paragraph(f"<b>Confidence Level:</b> {conf['level'].upper()}", 
                                  self.styles['Normal']))
            story.append(Paragraph(f"<i>{conf['message']}</i>", self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Summary
            story.append(Paragraph("Summary", self.styles['Heading3']))
            story.append(Paragraph(explainability['summary'], self.styles['Normal']))
        
        # Footer / Disclaimer
        story.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10
        )
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by an AI-powered decision support system "
            "and should not replace professional medical judgment. All predictions should be interpreted "
            "by qualified healthcare professionals in conjunction with clinical examination, patient history, "
            "and additional diagnostic tests. This tool is intended for research and clinical decision support only.",
            disclaimer_style
        ))
        
        # Build PDF
        doc.build(story)
        return output_path

