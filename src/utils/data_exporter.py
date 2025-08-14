import json
import csv
import io
import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataExporter:
    """Export automation results to various formats"""
    
    @staticmethod
    def to_csv(data: List[Dict], filename: str = None) -> Dict[str, Any]:
        """Export data to CSV format"""
        if not data:
            return {"error": "No data to export"}
        
        filename = filename or f"automation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            # Create CSV in memory
            output = io.StringIO()
            
            # Get all possible keys from all records
            all_keys = set()
            for item in data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            fieldnames = sorted(list(all_keys))
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for item in data:
                if isinstance(item, dict):
                    # Handle nested data by converting to string
                    row = {}
                    for key in fieldnames:
                        value = item.get(key, '')
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value)
                        else:
                            row[key] = str(value) if value is not None else ''
                    writer.writerow(row)
            
            csv_content = output.getvalue()
            output.close()
            
            return {
                "format": "csv",
                "filename": filename,
                "content": csv_content,
                "size_bytes": len(csv_content.encode('utf-8')),
                "rows": len(data)
            }
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return {"error": f"CSV export failed: {str(e)}"}
    
    @staticmethod
    def to_json(data: List[Dict], filename: str = None, pretty: bool = True) -> Dict[str, Any]:
        """Export data to JSON format"""
        filename = filename or f"automation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            if pretty:
                json_content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            else:
                json_content = json.dumps(data, ensure_ascii=False, default=str)
            
            return {
                "format": "json",
                "filename": filename,
                "content": json_content,
                "size_bytes": len(json_content.encode('utf-8')),
                "records": len(data)
            }
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return {"error": f"JSON export failed: {str(e)}"}
    
    @staticmethod
    def to_excel(data: List[Dict], filename: str = None) -> Dict[str, Any]:
        """Export data to Excel format (requires openpyxl)"""
        filename = filename or f"automation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        try:
            # Try to import openpyxl
            try:
                from openpyxl import Workbook
                from openpyxl.utils.dataframe import dataframe_to_rows
                import pandas as pd
            except ImportError:
                return {"error": "Excel export requires openpyxl and pandas libraries"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Create workbook
            wb = Workbook()
            ws = wb.active
            ws.title = "Automation Results"
            
            # Add data to worksheet
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
            
            # Save to memory
            excel_buffer = io.BytesIO()
            wb.save(excel_buffer)
            excel_content = excel_buffer.getvalue()
            excel_buffer.close()
            
            # Encode as base64 for transfer
            excel_b64 = base64.b64encode(excel_content).decode('utf-8')
            
            return {
                "format": "excel",
                "filename": filename,
                "content_base64": excel_b64,
                "size_bytes": len(excel_content),
                "rows": len(data)
            }
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return {"error": f"Excel export failed: {str(e)}"}
    
    @staticmethod
    def auto_export(session_data: Dict, requested_formats: List[str] = None) -> Dict[str, Any]:
        """Automatically export session data in requested formats"""
        if not requested_formats:
            requested_formats = ['json', 'csv']  # Default formats
        
        # Extract exportable data from session
        exportable_data = DataExporter._prepare_export_data(session_data)
        
        if not exportable_data:
            return {"error": "No exportable data found"}
        
        results = {}
        
        for format_type in requested_formats:
            if format_type.lower() == 'csv':
                results['csv'] = DataExporter.to_csv(exportable_data)
            elif format_type.lower() == 'json':
                results['json'] = DataExporter.to_json(exportable_data)
            elif format_type.lower() == 'excel':
                results['excel'] = DataExporter.to_excel(exportable_data)
            else:
                results[format_type] = {"error": f"Unsupported format: {format_type}"}
        
        return {
            "export_timestamp": datetime.now().isoformat(),
            "data_points": len(exportable_data),
            "formats": results
        }
    
    @staticmethod
    def _prepare_export_data(session_data: Dict) -> List[Dict]:
        """Prepare session data for export"""
        export_data = []
        
        # Extract step results
        step_results = session_data.get('execution_context', {}).get('step_results', [])
        variables = session_data.get('execution_context', {}).get('variables', {})
        
        for i, step_result in enumerate(step_results):
            export_row = {
                'step_number': i + 1,
                'step_id': step_result.get('step_id'),
                'result': step_result.get('result', ''),
                'completed_at': step_result.get('completed_at'),
                'analysis_status': step_result.get('analysis', {}).get('status'),
                'goal_progress': step_result.get('goal_progress', 0)
            }
            
            # Add extracted variables
            analysis = step_result.get('analysis', {})
            extracted_data = analysis.get('data_extracted', {})
            export_row.update(extracted_data)
            
            export_data.append(export_row)
        
        # Add overall session variables as a summary row
        if variables:
            summary_row = {
                'step_number': 'SUMMARY',
                'step_id': 'session_summary',
                'result': 'Final extracted variables',
                'completed_at': datetime.now().isoformat()
            }
            summary_row.update(variables)
            export_data.append(summary_row)
        
        return export_data
