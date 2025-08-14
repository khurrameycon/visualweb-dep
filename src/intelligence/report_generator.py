import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.utils.data_exporter import DataExporter

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generate intelligent analysis reports from automation results"""
    
    @staticmethod
    async def generate_analysis_report(session_data: Dict, llm, report_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate comprehensive analysis report using LLM"""
        
        try:
            # Prepare data for analysis
            analysis_data = ReportGenerator._prepare_analysis_data(session_data)
            
            if not analysis_data['steps']:
                return {"error": "No execution data found for analysis"}
            
            # Generate LLM-based analysis
            analysis_prompt = f"""Analyze this browser automation execution and provide insights:

EXECUTION SUMMARY:
- Total Steps: {analysis_data['total_steps']}
- Successful Steps: {analysis_data['successful_steps']}
- Failed Steps: {analysis_data['failed_steps']}
- Adaptations Made: {analysis_data['adaptations_count']}
- Goal Progress: {analysis_data['goal_progress']}%
- Duration: {analysis_data['duration_minutes']} minutes

MAIN GOAL: {analysis_data['main_goal']}

STEP DETAILS:
{json.dumps(analysis_data['step_summary'], indent=2)}

EXTRACTED DATA:
{json.dumps(analysis_data['variables'], indent=2)}

ADAPTATIONS:
{json.dumps(analysis_data['adaptations'], indent=2)}

Please provide a comprehensive analysis including:
1. Overall Performance Assessment
2. Success Factors and Failure Points
3. Data Quality and Insights
4. Efficiency Analysis
5. Recommendations for Improvement
6. Key Findings Summary

Format as structured markdown report."""

            response = await llm.ainvoke([{"role": "user", "content": analysis_prompt}])
            analysis_content = response.content
            
            # Structure the final report
            report = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "execution_metadata": analysis_data,
                "llm_analysis": analysis_content,
                "data_exports": None,  # Will be added if requested
                "recommendations": ReportGenerator._extract_recommendations(analysis_content)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"error": f"Report generation failed: {str(e)}"}
    
    @staticmethod
    def _prepare_analysis_data(session_data: Dict) -> Dict[str, Any]:
        """Prepare session data for LLM analysis"""
        
        execution_context = session_data.get('execution_context', {})
        step_results = execution_context.get('step_results', [])
        goal_tracker = getattr(session_data.get('goal_tracker'), '__dict__', {}) if session_data.get('goal_tracker') else {}
        
        # Calculate metrics
        total_steps = len(step_results)
        successful_steps = len([s for s in step_results if s.get('analysis', {}).get('status') == 'success'])
        failed_steps = len([s for s in step_results if s.get('analysis', {}).get('status') == 'failed'])
        adaptations = execution_context.get('adaptations_made', [])
        
        # Calculate duration
        if step_results:
            first_step = datetime.fromisoformat(step_results[0].get('completed_at', datetime.now().isoformat()))
            last_step = datetime.fromisoformat(step_results[-1].get('completed_at', datetime.now().isoformat()))
            duration_minutes = (last_step - first_step).total_seconds() / 60
        else:
            duration_minutes = 0
        
        # Prepare step summary
        step_summary = []
        for i, step in enumerate(step_results):
            analysis = step.get('analysis', {})
            step_summary.append({
                'step': i + 1,
                'status': analysis.get('status', 'unknown'),
                'what_happened': analysis.get('what_happened', '')[:100],
                'data_extracted': list(analysis.get('data_extracted', {}).keys()),
                'goal_progress_delta': analysis.get('goal_progress_delta', 0)
            })
        
        return {
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'failed_steps': failed_steps,
            'success_rate': (successful_steps / total_steps * 100) if total_steps > 0 else 0,
            'adaptations_count': len(adaptations),
            'goal_progress': goal_tracker.get('current_progress', 0),
            'main_goal': goal_tracker.get('main_goal', 'Unknown goal'),
            'duration_minutes': round(duration_minutes, 2),
            'step_summary': step_summary,
            'variables': execution_context.get('variables', {}),
            'adaptations': adaptations,
            'deviation_count': goal_tracker.get('deviation_count', 0)
        }
    
    @staticmethod
    def _extract_recommendations(analysis_content: str) -> List[str]:
        """Extract actionable recommendations from LLM analysis"""
        recommendations = []
        
        # Simple extraction - look for numbered lists or bullet points
        lines = analysis_content.split('\n')
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if 'recommendation' in line.lower() or 'suggest' in line.lower():
                in_recommendations = True
                continue
            
            if in_recommendations and (line.startswith('-') or line.startswith('*') or any(line.startswith(str(i)) for i in range(1, 10))):
                recommendation = line.lstrip('-*0123456789. ').strip()
                if len(recommendation) > 10:  # Filter out very short items
                    recommendations.append(recommendation)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    @staticmethod
    async def generate_report_with_exports(session_data: Dict, llm, export_formats: List[str] = None) -> Dict[str, Any]:
        """Generate analysis report with data exports"""
        
        # Generate main analysis
        report = await ReportGenerator.generate_analysis_report(session_data, llm)
        
        if report.get('error'):
            return report
        
        # Add data exports
        if export_formats:
            export_results = DataExporter.auto_export(session_data, export_formats)
            report['data_exports'] = export_results
        
        return report
