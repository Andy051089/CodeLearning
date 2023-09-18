package hello;

public class MultipleInt2 {
	public static void main(String[] args) {
		int[] scores = {59,67,73,85,96,100};
		System.out.println("數組的長度:"+scores.length);
		//變例
		for(int i = 0;i<scores.length;i++) {
			int score = scores[i];
			System.out.println(score);
		}
	}
}
