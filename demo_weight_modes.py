"""
Demo script to showcase all weight modes
Perfect for presentation/demo
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matcher.job_resume_matcher import match_resumes


def demo_all_modes(job_file):
    """
    Run matching with all different weight modes to show flexibility.
    """
    
    if not os.path.exists(job_file):
        print(f" Job file not found: {job_file}")
        return
    
    with open(job_file, 'r', encoding='utf-8') as f:
        job_description = f.read()
    
    print("\n" + "="*70)
    print(" RESUME SCREENING DEMO - ALL WEIGHT MODES")
    print("="*70)
    print(f"Job: {job_file}\n")
    
    # Test all modes
    modes = ['smart', 'balanced', 'skills', 'experience', 'embeddings']
    
    all_results = {}
    
    for mode in modes:
        print("\n" + "🔹"*35)
        print(f"\n Testing Mode: {mode.upper()}")
        print("🔹"*35)
        
        results = match_resumes(job_description, top_k=5, weight_mode=mode)
        all_results[mode] = results
        
        print("\n Top 5 Results:")
        print(results[['filename', 'final_score', 'matched_skills']].to_string(index=False))
        
        input("\nPress Enter to continue to next mode...")
    
    # Comparison summary
    print("\n" + "="*70)
    print(" COMPARISON SUMMARY")
    print("="*70)
    
    print("\n Top Candidate by Mode:")
    for mode, results in all_results.items():
        top = results.iloc[0]
        print(f"   {mode.upper():15} → {top['filename']:35} (Score: {top['final_score']:.4f})")
    
    print("\n Demo Complete!")
    print("\nKey Takeaway: System adapts weights based on job requirements")
    print("             or allows manual control for specific needs.")


def quick_demo(job_file):
    """
    Quick demo with just smart mode (for time-constrained presentations)
    """
    
    if not os.path.exists(job_file):
        print(f" Job file not found: {job_file}")
        return
    
    with open(job_file, 'r', encoding='utf-8') as f:
        job_description = f.read()
    
    print("\n" + "="*70)
    print("QUICK DEMO - SMART WEIGHT SELECTION")
    print("="*70)
    
    results = match_resumes(job_description, top_k=5, weight_mode='smart')
    
    print("\n" + "="*70)
    print(" TOP 5 MATCHES")
    print("="*70)
    print(results.to_string(index=False))
    
    print("\n Done! The system automatically adapted to this job type.")


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Full demo:  python demo_weight_modes.py <job_file.txt> full")
        print("  Quick demo: python demo_weight_modes.py <job_file.txt>")
        sys.exit(1)
    
    job_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'quick'
    
    if mode == 'full':
        demo_all_modes(job_file)
    else:
        quick_demo(job_file)